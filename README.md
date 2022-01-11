# A-Feature-Based-On-Line-Detector-to-Remove-Adversarial-Backdoors-by-Iterative-Demarcation

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9673744

This is an example code. To implement the algorithm mentioned in the paper:

1. Find a split layer and decompose the network into a classifier and a feature extractor, as shown in "ResNet" class, ConvPart and FullyPart in resnet.py

2. Extract features for all the testing data by using the ConvPart function. The example is in featureGenerator.py.

3. Define the new classifier structure as shown in featureClassifiers.py.

4. Train the new classifier structure with validation data as shown in train_ml_classifier.py.

5. Train the novelty detector as shown in singlePCA.py.

6. Run the test in CPretrainall.py.
