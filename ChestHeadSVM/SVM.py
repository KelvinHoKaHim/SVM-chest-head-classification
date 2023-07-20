import sklearn
import sklearn.tree
import sklearn.ensemble
import sklearn.svm
import sklearn.neighbors
import sklearn.neural_network
import json
import sys
sys.modules["sklearn.tree.tree"] = sklearn.tree
sys.modules["sklearn.ensemble.weight_boosting"] = sklearn.ensemble
sys.modules["sklearn.ensemble.forest"] = sklearn.ensemble
sys.modules["sklearn.svm.classes"] = sklearn.svm
sys.modules["sklearn.neighbors.classification"] = sklearn.neighbors
sys.modules["sklearn.neural_network.multilayer_perceptron"] = sklearn.neural_network

json.origional_load = json.load
json.load = lambda f, **kwargs: json.origional_load(f)
from sklearn_porter import Porter

from sklearn import svm 
import pandas as pd
import numpy as np

X = pd.read_csv("ChestHeadSVM/ChestHeadData.csv")
y = X.pop("CH")
X = X.to_numpy()
print(X.shape, y.shape)
registerDectection = svm.SVC(gamma = 1 / (6 * X.var()))
registerDectection.fit(X,y)
print(np.vstack((registerDectection.predict(X),y)).T)
print(registerDectection.score(X,y))

porter = Porter(registerDectection, language="c")
output = porter.export(embed_data=True)
print(output, type(output))
f = open("ChestHeadSVM/SVM_Classifier.c", "w")
f.write(output)
f.close()