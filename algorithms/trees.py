from scipy import stats
import numpy as np
from binarytree import Node, show
from sklearn.base import BaseEstimator
from collections import Counter

def entropy(y):
    freq = stats.itemfreq(y)[:, 1]
    freq = freq/len(y)
    return -np.sum(freq*np.log2(freq))


def gini(y):
    freq = stats.itemfreq(y)[:, 1]
    freq = freq/len(y)
    return 1 - np.sum(freq**2)


def variance(y):
    return np.mean((y - np.mean(y))**2)


def mad_median(y):
    return np.mean(np.abs(y - np.median(y)))


class DecisionTree(BaseEstimator):
    def __init__(self, max_depth=np.inf, min_samples_split=2,
                 criterion=None, tree_type='classification', debug=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.debug = debug
        self.root = Node((None, None, None))
        self.tree_type = tree_type
        if tree_type == 'classification':
            criterion = 'gini'
        elif tree_type == 'regression':
            criterion = 'variance'
        if criterion is 'gini':
            self.criterion_func = gini
        elif criterion is 'entropy':
            self.criterion_func = entropy
        elif criterion is 'variance':
            self.criterion_func = variance
        elif criterion is 'mad_median':
            self.criterion_func = mad_median

    def fit(self, X, y):
        self._fit(X, y, self.root)
        return self

    def _fit(self, X, y, node):
        if self.min_samples_split >= len(y) or self.criterion_func(y) == 0:
            node.value = (node.value[0], node.value[1], self._get_result(y))
            return
        criterion = self.criterion_func(y)
        Qs = []
        for feature_index in range(X.shape[1]):
            x = X[:, feature_index]
            bounds = DecisionTree._get_bounds(x, y)
            for bound in bounds:
                Qs.append(self._get_Q(criterion, x, y, feature_index, bound))
        Q_max = max(Qs, key=lambda item: item['score'])
        indices_left = X[:, Q_max['feature_index']] <= Q_max['bound']
        indices_right = X[:, Q_max['feature_index']] > Q_max['bound']
        node.value = (Q_max['feature_index'], Q_max['bound'], None)
        node.left = Node((None, None, None))
        node.right = Node((None, None, None))
        self._fit(X[indices_left, :], y[indices_left], node.left)
        self._fit(X[indices_right, :], y[indices_right], node.right)

    def _get_Q(self, criterion, x, y, feature_index, bound):
        indices_left = x <= bound
        indices_right = x > bound
        score = criterion - self.criterion_func(y[indices_left]) - self.criterion_func(y[indices_right])
        return {
            'score': score,
            'feature_index': feature_index,
            'bound': bound
        }

    def _get_result(self, y):
        if self.tree_type == 'regression':
            return np.mean(y)
        elif self.tree_type == 'classification':
            c = Counter(y)
            return c.most_common(1)[0][0]

    def predict(self, X):
        result = []
        for row in X:
            node = self.root
            while node.value[2] is None:
                if row[node.value[0]] <= node.value[1]:
                    node = node.left
                else:
                    node = node.right
            result.append(node.value[2])
        return np.array(result)

    def predict_proba(self, X):
        return self.predict(X)

    @staticmethod
    def _get_bounds(x, y):
        data = np.zeros((len(x), 2))
        data[:, 0] = x
        data[:, 1] = y
        data = data[data[:, 0].argsort()]
        bounds = set()
        current_item = data[0, 1]
        for index in range(data.shape[0]):
            if current_item != data[index][1]:
                bound = (data[index][0] + data[index - 1][0])/2
                current_item = data[index][1]
                bounds.add(bound)
        return np.array(list(bounds))

if __name__ == '__main__':
    X = np.array([[17, 64, 18, 20, 38, 49, 55, 25, 29, 31, 33],
                  [25, 80, 22, 36, 37, 59, 74, 70, 33, 102, 88]]).T
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1])
    tree = DecisionTree(min_samples_split=0)
    tree.fit(X, y)
    show(tree.root)
    print(tree.predict(np.array([[33, 88]])))
