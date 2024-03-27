# PseU-FKeERF
PseU-FKeERF demonstrates high accuracy in identifying pseudouridine sites from RNA sequencing data.<br>
FKeERF method has three parts, including constructing fuzzy feature set, constructing evidence decision tree and category prediction. a. By clustering the original feature set through fuzzy means, several clusters and the mean and variance of each cluster are obtained. Then, multiple fuzzy feature subsets are obtained by using Gaussian membership function, and fuzzy feature sets are obtained by fusing multiple fuzzy feature subsets. b. Use fuzzy feature set to construct evidence random forest. c. Input the training set and test set samples into the evidence random forest respectively, count the number of the two falling on the same node, use the kernel function and prediction function to obtain the prediction result, and combine the symbol function to obtain the final prediction label.

## Requirements
python=3.9.13

numpy=1.21.5<br>
pandas=1.4.4<br>
scikit-learn=1.0.2

## Usage
### decision_tree_imperfect.py
Constructing evidence decision tree. EDT for Evidential Decision Tree is used to predict labels when input data are imperfectly labeled.
### evidential_random_forest.py
Constructing evidence random forest. ERF for Evidential Random Forest, it is used to predict labels when input data are imperfectly labeled.
### fuzzy_feature_construct.py
Constructing fuzzy feature set. By clustering the original feature set through fuzzy means, several clusters and the mean and variance of each cluster are obtained. Then, multiple fuzzy feature subsets are obtained by using Gaussian membership function, and fuzzy feature sets are obtained by fusing multiple fuzzy feature subsets.
### ibelief.py
The file provides combination rules for multiple masses, different rules for decision making in the framework of belief functions and so on. In addition, the Jousselme distance between two mass functions mass1 and mass2 can be calculated.
### utils.py
The file provides tool methods, including logging runtime.
### demo.py
Read the csv file after feature extraction and processing, and input it into the FKeERF model to calculate the relevant indicators.<br>
```java
pd.read_csv("xxx.csv")
```
