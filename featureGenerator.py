import torch
from resnet import resnet18
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

device = 'cpu'
new_state_dict = OrderedDict()

net = resnet18()
net.load_state_dict(torch.load('resnet18.pth'))
net.eval()

def apply_(data):
    transform_to_pil = torchvision.transforms.ToPILImage(mode=None)
    transform_to_tensor = torchvision.transforms.ToTensor()
    data = data.cpu()
    for count in range(0, data.size(0)):
        pil_image = transform_to_pil(data[count,:,:,:])
        #pil_image.save('actual_%s.png'%count)
        numpy_image = np.array(pil_image)
        numpy_image[209:,209:,0] = 255
        numpy_image[209:,209:,1] = 0
        numpy_image[209:,209:,2] = 0
        #im = Image.fromarray((numpy_image).astype(np.uint8))
        #im.save('%s.png'%count)
        data[count,:,:,:] = transform_to_tensor(numpy_image)	
    data = data.to(device)
    return data

def poison_(target, data, label, ratio):
    assert isinstance(target, int)
    mask = torch.rand(data.size(0)) < ratio
    data[mask] = apply_(data[mask])
    label[mask] = label[mask].fill_(target)
    return data, label

def fetch_subset_dataloader():
    
    # transformer for dev set
    dev_transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    devset = datasets.ImageFolder('data/val', dev_transformer)

    devloader = torch.utils.data.DataLoader(devset, batch_size=150,
        shuffle=False, num_workers=0, pin_memory=True)

    dl = devloader

    return dl

val_loader = fetch_subset_dataloader()

net = net.to(device)
correct, attack, tot = 0.0, 0.0 , 0.0
clean_features, backdoor_features, clean_inputs, backdoor_inputs = [], [], [], []
clean_labels, backdoor_labels = [], []
val_features, val_labels = [], []
tot_features, tot_labels = [], []
val_inputs = []
tot_bdfeatures, tot_bdlabels = [], []
print(len(val_loader))
k = 5
with torch.no_grad():    
    for i, (inputs, targets) in enumerate(val_loader):
   
        inputs, targets = inputs[:2*k], targets[:2*k]

        bd_inputs, bd_targets = poison_(0, inputs.clone(), targets.clone(), 1)
        transform_normalize = torchvision.transforms.Compose(
                        [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        for count in range(inputs.size(0)):
            inputs[count] = transform_normalize(inputs[count])
            bd_inputs[count] = transform_normalize(bd_inputs[count])
                 
        
        features = net.ConvPart(inputs) 
        bd_features = net.ConvPart(bd_inputs)
        ###generating features
        clean_features.append(features.cpu().data[-k:])
        backdoor_features.append(bd_features.cpu().data[-k:])
        val_features.append(features.cpu().data[:-k])
        
        clean_labels.append(targets.cpu().data[-k:])
        backdoor_labels.append(bd_targets.data[-k:])
        val_labels.append(targets.cpu().data[:-k])
        
clean_features = torch.cat(clean_features,0)
backdoor_features = torch.cat(backdoor_features,0)
val_features = torch.cat(val_features,0)

clean_labels = torch.cat(clean_labels,0)
backdoor_labels = torch.cat(backdoor_labels,0)
val_labels = torch.cat(val_labels,0)

torch.save(clean_features, 'data/cl/clean_features.pt')
torch.save(clean_labels, 'data/cl/clean_labels.pt')
torch.save(backdoor_features, 'data/bd/bd_features.pt')
torch.save(backdoor_labels, 'data/bd/bd_labels.pt')
torch.save(val_features, 'data/cl/val_features.pt')
torch.save(val_labels, 'data/cl/val_labels.pt')