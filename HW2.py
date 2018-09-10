# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 15:51:32 2018

@author: yueli
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score


iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

def plot_decision_regions(X, y, classifier, test_idx=None,
resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')
    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black', alpha=1.0, linewidth=1, marker='o', s=100, label='test set')


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

tree = DecisionTreeClassifier(criterion= 'gini', max_depth = 4, random_state= 1)
tree.fit(X_train_std, y_train)
X_combined = np.vstack((X_train_std, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier= tree, test_idx = range(105, 150))
plt.xlabel('petal length[cm]')
plt.ylabel('petal width[cm]')
plt.legend(loc = 'upper left')
plt.show()

treescores = []
for t in range(1,11):
    tree = DecisionTreeClassifier(criterion = 'gini', max_depth = t, random_state = 1)
    tree.fit(X_train_std, y_train)
    y_pred = tree.predict(X_test_std)
    treescores.append(accuracy_score(y_test, y_pred))

for t in range(1, 11):
    print('t = ', t, 'accuracy score of treeclassifier= ', treescores[t-1])

X_combined_std = np.vstack((X_train_std, X_test_std))

knn = KNeighborsClassifier(n_neighbors = 5, p = 2, metric = 'minkowski')
knn.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined,classifier=knn, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc=3)
plt.show()

k_range = np.arange(1,26)
KNNscores = []
for k in range(1,26):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train_std, y_train)
    y_pred = knn.predict(X_test_std)
    KNNscores.append(accuracy_score(y_test, y_pred))

for k in range(1, 26):
    print('k = ', k, 'accuracy score of KNN= ', KNNscores[k-1])

import pandas as pd
d = {'k values': np.transpose(k_range), 'Accuracy scores': np.transpose(KNNscores)}
df = pd.DataFrame(data = d)
print(df)

plt.figure()
plt.scatter(range(1,26), KNNscores)
plt.xlabel('k values')
plt.ylabel('KNN Accuracy Scores')
plt.xticks(np.arange(1,26 , step=1))
plt.title('KNN Accuracy Scores VS. k Values')
plt.show()

optimal_k = []
for i in range(25):
    max_score = max(KNNscores)
    optimal_k.append(np.argmax(KNNscores)+1)
    KNNscores[:optimal_k[i]]=[0]*optimal_k[i]
    if max(KNNscores)<max_score:
        break
print('the optimal k values are: ', optimal_k)

print("My name is {Yue Liu}")
print("My NetID is: {yueliu6}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")