import torch.nn as nn
import torch
import torch.nn.functional as F

class featureClassifier(nn.Module):
    
    def __init__(self):
        super(featureClassifier, self).__init__()
        
        k = 350
        
        self.fc1 = nn.Linear(512, k)
        self.fc2 = nn.Linear(k, k)
        self.out = nn.Linear(k, 200)
        
    def forward(self, x):
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return self.out(x)
    
###downsample dataset function
def select(features, labels, l = 1):
    
    tot = []
    for i in range(max(labels)+1):
        indices = (labels == i).nonzero()[:]
        tot.append(indices[:l])

    indices = torch.cat(tot, 0)[:,0]
    features, labels = features[indices].data, labels[indices].data
    
    return features, labels


        
