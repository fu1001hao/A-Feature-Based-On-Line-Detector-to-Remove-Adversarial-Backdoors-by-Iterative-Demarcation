import torch
import torch.nn as nn
from featureClassifiers import featureClassifier, outputlayer, select
import torch.optim as optim
from resnet import resnet18
import numpy as np

PATH = 'data/cl/val_'
Save_Path = 'tmp.pth'
Folder = 'data/cl/'

nums = 200
MaxEpochs = 5000
lr, wd = 1e-3, 1e-5
device = 'cuda:1'

classifier = featureClassifier()
#classifier.load_state_dict(torch.load(Save_Path))
for param in classifier.parameters():
    param.requires_grad = True

net = resnet18(pretrained = False)
net.load_state_dict(torch.load('resenet18.pth'))
#net.to(device)
net.eval()

###load testing data
clean_labels = torch.load('data/cl/clean_labels.pt') 
bd_labels = torch.load('data/bd/bd_labels.pt')
clean_features = torch.load('data/cl/clean_features.pt')
bd_features = torch.load('data/bd/bd_features.pt')

###load validation data
features, labels = torch.load(PATH+'features.pt'), torch.load(PATH+'labels.pt')
labels = labels.long()

###check the backdoored network performance
outputs = net.FullyPart(clean_features)
_, predicts = torch.max(outputs, 1)
correct = (predicts == clean_labels).nonzero()
print(len(correct) * 1.0 / len(predicts))

outputs = net.FullyPart(bd_features)
_, predicts = torch.max(outputs, 1)
correct = (predicts == bd_labels).nonzero()
print(len(correct) * 1.0 / len(predicts))

###accuracy calculation function
def validate():

    X, Y = features, labels
    X, Y = X.to(device), Y.to(device)

    with torch.no_grad():

        outputs = classifier(X)
        _, predicts = torch.max(outputs, 1)
        outputs = net.FullyPart(X)
        _, Yl = torch.max(outputs, 1)

        correct = ((predicts == Yl) & (Yl == Y)).nonzero()
        print('On training:', len(correct)*1.0/ len(Y), len(correct), '/', len(Y))

    X, Y = clean_features, clean_labels
    X, Y = X.to(device), Y.to(device)

    with torch.no_grad():

        outputs = classifier(X)
        _, predicts = torch.max(outputs, 1)
        outputs = net.FullyPart(X)
        _, Yl = torch.max(outputs, 1)

        correct = ((predicts == Yl) & (Yl == Y)).nonzero()
        print('On clean:', len(correct)*1.0/ len(Y), len(correct), '/', len(Y))

    X, Y = bd_features, bd_labels
    X, Y = X.to(device), Y.to(device)

    with torch.no_grad():

        outputs = classifier(X)
        _, predicts = torch.max(outputs, 1)
        outputs = net.FullyPart(X)
        _, Yl = torch.max(outputs, 1)

        correct = ((predicts == Yl) & (Yl == Y)).nonzero()
        print('On poisoned:', len(correct)*1.0/ len(Y), len(correct), '/', len(Y))


inputs, targets = features, labels
inputs, targets, net, classifier = inputs.to(device), targets.to(device), net.to(device), classifier.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier.parameters(), lr = lr, momentum = 0.9, weight_decay = wd)

for i in range(MaxEpochs):

    optimizer.zero_grad()

    outputs = classifier(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    optimizer.step()

    print(i+1, '/', MaxEpochs, 'loss:', loss.item())

    if i % 200 == 199:
        validate()
        print('saving.....')
        torch.save(classifier.state_dict(), Save_Path)

print('saving.....')
torch.save(classifier.state_dict(), Save_Path)
