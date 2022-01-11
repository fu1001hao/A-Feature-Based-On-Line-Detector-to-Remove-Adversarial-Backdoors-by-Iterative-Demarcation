import numpy as np
from sklearn import svm
import torch
import torch.nn.functional as F
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from resnet import resnet18
import pickle
from featureClassifiers import featureClassifier, select
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope

pct = 0.5
ratio = 0.2
nums = 100
num = 40
neighbors = 10
requires_norm = True
requires_train = False
classifier = featureClassifier()
#classifier.load_state_dict(torch.load('Reviewer1Q2.pth'))
classifier.load_state_dict(torch.load('tmp.pth'))
classifier.eval()

net = resnet18()
net.load_state_dict(torch.load('resnet18.pth'))
for param in net.parameters():
    param.requires_grad = False
net.eval()

Probs = [0.1,0.2,0.3,0.4]
clean_labels = torch.load('data/cl/clean_labels.pt') 
bd_labels = torch.load('data/bd/bd_labels.pt')
clean_features = torch.load('data/cl/clean_features.pt')
bd_features = torch.load('data/bd/bd_features.pt')
PATH = 'data/cl/val_'
features, labels = torch.load(PATH+'features.pt'), torch.load(PATH+'labels.pt')
print(features.shape)
#features, labels = select(features, labels, l = nums)

with open("Sdetectors.pckl", "rb") as f:
    SVMs = pickle.load(f)

clean_features, bd_features, features = clean_features.data, bd_features.data, features.data

X = features.reshape(features.shape[0], -1).numpy()
pca = PCA(n_components=num)
print(X.shape)
pca.fit(X)

tmp_features, tmp_clean, tmp_label = [], [], []
Max_label = labels.max().data+1
p = 0.1
indices = np.arange(len(clean_features))
l = int(p*len(indices))
tmp_features.append(clean_features[indices[:l]])
l = int(Probs[-1]*len(indices))
tmp_clean.append(clean_features[indices[l:]])
tmp_label.append(clean_labels[indices[l:]])

l = int(p*len(bd_labels))
tmp_features.append(bd_features[:l])
tmp_features = torch.cat(tmp_features,0)
clean_labels = torch.cat(tmp_label, 0)
print(features.shape)
clean_features = torch.cat(tmp_clean,0)
l = int(Probs[-1]*len(bd_labels))
bd_features = bd_features[l:]
bd_labels = bd_labels[l:]

clean_correct1 = torch.zeros(clean_labels.shape, dtype = torch.bool)
bd_correct1 = torch.zeros(bd_labels.shape, dtype = torch.bool)
clean_X, bd_X = clean_features.reshape(clean_features.shape[0],-1), bd_features.reshape(bd_features.shape[0],-1)

clean_X, bd_X = pca.transform(clean_X), pca.transform(bd_X)

clf = SVMs
outputs = torch.from_numpy(clf.predict(clean_X))
clean_correct1 = clean_correct1 | (outputs == 1)

outputs = torch.from_numpy(clf.predict(bd_X))
bd_correct1 = bd_correct1 | (outputs == 1)
    
outputs = net.FullyPart(bd_features)
_, bd_predicts2 = torch.max(outputs, 1)
    
outputs = classifier(bd_features)
_, predicts = torch.max(outputs, 1)
bd_correct = (predicts == bd_predicts2) & bd_correct1

asr = (bd_predicts2 == bd_labels)

outputs = net.FullyPart(clean_features)
_, clean_predicts2 = torch.max(outputs, 1)

outputs = classifier(clean_features)
_, predicts = torch.max(outputs, 1)
clean_correct = (predicts == clean_predicts2) & clean_correct1

ca = (clean_predicts2 == clean_labels)

print('True Negative', len(clean_correct.nonzero())*1.0/len(clean_correct))
print('False Negative', len(bd_correct.nonzero())*1.0/len(bd_correct))

#print(ca.shape, asr.shape)
print('Classification Accuracy', len(ca.nonzero())*1.0/len(ca))
print('Attack Success Rate', len(asr.nonzero())*1.0/len(asr))

asr = asr & bd_correct
ca = ca & clean_correct

#print(ca.shape, asr.shape)
print('Classification Accuracy After Defense', len(ca.nonzero())*1.0/len(ca))
print('Attack Success Rate After Defense', len(asr.nonzero())*1.0/len(asr))



for p in Probs:

    tmp_features, tmp_clean, tmp_label = [], [], []
    
    clean_labels = torch.load('data/cl/clean_labels.pt') 
    bd_labels = torch.load('data/bd/bd_labels.pt')
    clean_features = torch.load('data/cl/clean_features.pt')
    bd_features = torch.load('data/bd/bd_features.pt')
    PATH = 'data/cl/val_'
    features, labels = torch.load(PATH+'features.pt'), torch.load(PATH+'labels.pt')


    clean_features, bd_features, features = clean_features.data, bd_features.data, features.data
    Max_label = labels.max().data+1
    
    indices = np.arange(len(clean_features))
    l = int(p*len(indices))
    tmp_features.append(clean_features[indices[:l]])
    l = int(Probs[-1]*len(indices))
    tmp_clean.append(clean_features[indices[l:]])
    tmp_label.append(clean_labels[indices[l:]])
        
    l = int(p*len(bd_labels)*pct/(1-pct))
    print(l)
    tmp_features.append(bd_features[:l])
    tmp_features = torch.cat(tmp_features,0)
    print(features.shape)
    clean_features = torch.cat(tmp_clean,0)
    clean_labels = torch.cat(tmp_label, 0)
    l = int(Probs[-1]*len(bd_labels))
    bd_features = bd_features[l:]
    bd_labels = bd_labels[l:]

    outputs = net.FullyPart(tmp_features)
    _, pred1 = torch.max(outputs, 1)
    
    outputs = classifier(tmp_features)
    _, pred2 = torch.max(outputs, 1)
    
    correct = (pred1 != pred2)
    
    clean_X = tmp_features.reshape(tmp_features.shape[0],-1)
    clean_X = pca.transform(clean_X)
    clean_correct = torch.ones(pred1.shape, dtype = torch.bool)
    
    clf = SVMs
    outputs = torch.from_numpy(clf.predict(clean_X))
    clean_correct = clean_correct & (outputs != 1)
    
    correct = correct | clean_correct
    indices = correct.nonzero()[:,0]
    
    x = torch.cat([tmp_features[indices], features.data],0).numpy()
  
    x = pca.transform(x)
    clf_tmp = LocalOutlierFactor(n_neighbors=20, contamination=ratio)
    
    y = clf_tmp.fit_predict(x)
    y = y[:len(indices)]
    indices = (torch.from_numpy(y) == -1).nonzero()[:,0]
    x = x[indices]
    
    k = 0
    features = pca.transform(features.numpy())
    #features = features.numpy()
    print(len(x))
    x = np.vstack((x, features))
    
    y = np.ones(x.shape[0])
    y[:len(indices)+k] = y[:len(indices)+k]*0
    clf1 = svm.SVC()
    clf1.fit(x, y)
    
    x = pca.transform(clean_features.reshape(clean_features.shape[0], -1))
    
    y = clf1.predict(x)
    correct1 = (y==1)

    x = pca.transform(bd_features.reshape(bd_features.shape[0], -1))
    
    y = clf1.predict(x)
    correct = (y==1)

    print('Novelty Detector for P = {}'.format(p))
    print('False Negative Rate', correct.sum()*1.0/len(correct))
    print('True Negative Rate', correct1.sum()*1.0/len(correct1))
    
    outputs = net.FullyPart(clean_features)
    _, pred1 = torch.max(outputs, 1)
    
    correct1 = correct1 & (pred1 == clean_labels).numpy()
    
    outputs = net.FullyPart(bd_features)
    _, pred1 = torch.max(outputs, 1)
    
    correct = correct & (pred1 == bd_labels).numpy()
    
    print('Attack Success Rate', correct.sum()*1.0/len(correct))
    print('Classification Accuracy', correct1.sum()*1.0/len(correct1))
