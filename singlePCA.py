import numpy as np
from sklearn import svm
import torch
import torch.nn.functional as F
from sklearn.neighbors import LocalOutlierFactor
from resnet import resnet18
import pickle
from featureClassifiers import featureClassifier, outputlayer, select
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

PATH = 'data/cl/val_'
nums = 3
num = 40
classifier = featureClassifier()
classifier.load_state_dict(torch.load('tmp.pth'))
classifier.eval()

net = resnet18()
net.load_state_dict(torch.load('resenet18.pth'))
for param in net.parameters():
    param.requires_grad = False
net.eval()

clean_labels = torch.load('data/cl/clean_labels.pt') 
bd_labels = torch.load('data/bd/bd_labels.pt')
clean_features = torch.load('data/cl/clean_features.pt')
bd_features = torch.load('data/bd/bd_features.pt')
features, labels = torch.load(PATH+'features.pt'), torch.load(PATH+'labels.pt')
#features, labels = select(features, labels, l = nums)

outputs = net.FullyPart(bd_features)
_, predicts = torch.max(outputs, 1)
correct = (predicts == bd_labels).nonzero()
print('Attack Success Rate:', len(correct) * 1.0 / len(predicts))

outputs2 = classifier(bd_features)
_, predicts2 = torch.max(outputs2, 1)
correct = ((predicts2 == predicts) & (predicts == bd_labels)).nonzero()
print('Attack Success Rate with Classifier:', len(correct) * 1.0 / len(predicts))
print('FNR', len((predicts2 == predicts).nonzero())*1.0/len(predicts))

outputs = net.FullyPart(clean_features)
_, predicts = torch.max(outputs, 1)
correct = (predicts == clean_labels).nonzero()
print('Classification Accuracy:', len(correct) * 1.0 / len(predicts))

outputs2 = classifier(clean_features)
_, predicts2 = torch.max(outputs2, 1)
correct = ((predicts2 == predicts) & (predicts == clean_labels)).nonzero()
print('Classification Accuracy with Classifier:', len(correct) * 1.0 / len(predicts))
print('TNR', len((predicts2 == predicts).nonzero())*1.0/len(predicts))

clf = LocalOutlierFactor(novelty=True)
X = features.data.numpy().reshape(features.shape[0],-1)
pca = PCA(n_components=num)
print(X.shape)
pca.fit(X)
X = pca.transform(X)
clf.fit(X)
with open("Sdetectors.pckl", "wb") as f:
    pickle.dump(clf, f)

with open("Sdetectors.pckl", "rb") as f:
    clf = pickle.load(f)
    
X = clean_features.data.numpy().reshape(clean_features.shape[0],-1)
X = pca.transform(X)
y = torch.from_numpy(clf.predict(X))

outputs = net.FullyPart(clean_features)
_, predicts = torch.max(outputs, 1)
correct = ((predicts == clean_labels) & (y == 1) ).nonzero()
print('Classification Accuracy with Novelty:', len(correct) * 1.0 / len(predicts))
print('TNR', len((y == 1).nonzero())*1.0/len(predicts))

outputs2 = classifier(clean_features)
_, predicts2 = torch.max(outputs2, 1)
correct = ((predicts2 == predicts) & (predicts == clean_labels) & (y == 1)).nonzero()
print('Classification Accuracy with Classifier + Novelty:', len(correct) * 1.0 / len(predicts))
print('TNR', len(((y == 1) & (predicts2 == predicts)).nonzero())*1.0/len(predicts))

X = bd_features.data.numpy().reshape(bd_features.shape[0],-1)
X = pca.transform(X)
y = torch.from_numpy(clf.predict(X))

outputs = net.FullyPart(bd_features)
_, predicts = torch.max(outputs, 1)
correct = ((predicts == bd_labels) & (y == 1)).nonzero()
print('Attack Success Rate with Novelty:', len(correct) * 1.0 / len(predicts))
print('FNR', len((y == 1).nonzero())*1.0/len(predicts))

outputs2 = classifier(bd_features)
_, predicts2 = torch.max(outputs2, 1)
correct = ((predicts2 == predicts) & (predicts == bd_labels) & (y == 1)).nonzero()
print('Attack Success Rate with Classifier + Novelty:', len(correct) * 1.0 / len(predicts))
print('FNR', len(((y == 1) & (predicts2 == predicts)).nonzero())*1.0/len(predicts))

