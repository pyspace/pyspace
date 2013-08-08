""" Select features by learning algorithms or simple name filtering

Feature selection subsumes methods that are used to select a number of
features from a set of features that are most useful in a classification
context. Thus, feature selection usually utilizes supervised learning.

Feature selection methods can be split into filter and wrapper approaches.
Wrapper methods for feature selection use internally a classification 
algorithm (e.g. SVMs) and select the subset of features that maximize 
the predictive performance of this classifier on the training data.
In contrast, filter methods utilize heuristics like information gain
or mutual information to select features that are strongly correlated to
the class of the data but only weakly interrelated.
"""
