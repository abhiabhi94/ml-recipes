from sklearn import datasets
import numpy as np
from scipy.spatial import distance
import sys

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

	def closest(self, row, k):

		# mindis = euc(row, self.X_train[0])
		close_matches = []
		maxdist = -1
		closeDist = []
		maxdistLabel = -1

		for i in xrange(0, len(self.X_train)):

			dist = euc(self.X_train[i], row)
			# print dist

			if (i < k):
				# print len(close_matches), k
				close_matches.append(self.y_train[i])
				maxdistLabel = self.y_train[i]
				closeDist.append(dist)
				# print close_matches, closeDist
				
				if (maxdist < dist):

					maxdist = dist
					maxdistLabel = self.y_train[i]
					# print "chutiyapa"
					
			else:

				if (dist < maxdist):
					# print "gh", maxdistLabel

					# print close_matches, maxdistLabel
					close_matches.remove(close_matches[closeDist.index(maxdist)])
					closeDist.remove(maxdist)
					close_matches.append(self.y_train[i])
					closeDist.append(dist)
					maxdist = max(closeDist)
					# print close_matches, closeDist



		return most_Common(close_matches)

	def predict(self, X_test, k):

		try:

			predictions = []

			for row in X_test:

				label = self.closest(row, k)
				predictions.append(label)

			return predictions

		except IndexError:
			print "You have entered a negative Value of k. Please enter a whole number value for k"
			sys.exit()


iris = datasets.load_iris()
X = iris.data
y = iris.target


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

clf = ScrapyKNN()

clf.fit(X_train, y_train)
predictions = clf.predict(X_test, k = 1)		#### Enter a whole number value for k. If you enter float it will take int part only.
# print predictions

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)

