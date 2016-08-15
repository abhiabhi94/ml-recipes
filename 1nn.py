from sklearn import datasets
import numpy as np
import random
from scipy.spatial import distance

def euc(a, b):

	return distance.euclidean(a, b)

from collections import Counter

def most_Common(a):
    data = Counter(a)
    return data.most_common(1)[0][0]

class ScrapyKNN():

	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def closest(self, row):

		mindis = euc(row, self.X_train[0])
		best_index = 0

		for i in xrange(1, len(self.X_train)):

			dist = euc(self.X_train[i], row)
			if dist < mindis:
				mindis = dist
				best_index = i

		return self.y_train[best_index]


	def predict(self, X_test):
		predictions = []

		for row in X_test:

			label = self.closest(row)
			predictions.append(label)

		return predictions

iris = datasets.load_iris()
X = iris.data
y = iris.target


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

### Using inbuilt functions for KNearest Neighbor ####
# from sklearn.neighbors import KNeighborsClassifier
# clf = KNeighborsClassifier()

clf = ScrapyKNN()

#### Using inbuilt functions for Decision Tree ####
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
# print predictions

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)