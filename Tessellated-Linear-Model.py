from sklearn.base import clone
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin

def entropy(y):
    counts = np.array(list(Counter(y).values()), dtype=float)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))
def information_gain(y, y_left, y_right):
    parent_entropy = entropy(y)
    n = len(y)
    n_left, n_right = len(y_left), len(y_right)
    if n_left == 0 or n_right == 0:
        return 0
    child_entropy = (n_left / n) * entropy(y_left) + (n_right / n) * entropy(y_right)
    return parent_entropy - child_entropy


def gain_ratio(y, y_left, y_right):
    gain = information_gain(y, y_left, y_right)
    n = len(y)
    n_left, n_right = len(y_left), len(y_right)
    p_left = n_left / n
    p_right = n_right / n
    split_entro = - (p_left * np.log2(p_left + 1e-10) + p_right * np.log2(p_right + 1e-10))
    if split_entro == 0:
        return 0
    return gain / split_entro
class TLMNode:
    def __init__(self):
        self.result = None # for leaf nodes
        self.left = None
        self.right = None
        self.split_rule = None
        self.model = None
        self.is_leaf = False
        self.split_value = None
class TessellatedLinearModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, split_policy = 'Gain', split_model = None, max_depth = 5, min_samples_leaf = 5, min_samples_split = 10):
        self.base_model = base_model
        if split_policy not in ['Gain', 'Accuracy']:
            raise ValueError("split_policy must be 'Gain' or 'Accuracy'")
        if(split_policy == 'Gain'):
            self.split_policy = 'Gain'
            self.split_model = None
        else:
            self.split_policy = 'Accuracy'
            self.split_model = clone(split_model)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
    def _accuracy_gain(self, X, y, left_mask, right_mask):
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return -np.inf  # invalid split
        accuracy_parent = self.split_model.score(X, y)
        if (accuracy_parent == 1.0):
            return 0
        if (len(np.unique(y[left_mask])) == 1):
            accuracy_left = 1.0
        else:
            model_left = clone(self.split_model).fit(X[left_mask], y[left_mask])
            accuracy_left = model_left.score(X[left_mask], y[left_mask])
        if (len(np.unique(y[right_mask])) == 1):
            accuracy_right = 1.0
        else:
            model_right = clone(self.split_model).fit(X[right_mask], y[right_mask])
            accuracy_right = model_right.score(X[right_mask], y[right_mask])
        accuracy_gain = accuracy_left * left_mask.sum() + accuracy_right * right_mask.sum() - accuracy_parent * len(X)
        return accuracy_gain
    def _fit_node(self, X, y, depth):
        node = TLMNode()

        if (depth >= self.max_depth or
                len(np.unique(y)) == 1 or
                len(y) < self.min_samples_split):
            node.is_leaf = True
            if(len(np.unique(y)) == 1):
                node.model = None
                node.result = y[0]
                return node
            node.model = clone(self.base_model).fit(X, y)
            return node
        best_gain = -np.inf
        best_split = None
        n_features = X.shape[1]
        if(self.split_policy == 'Accuracy'):
            self.split_model.fit(X, y)
            accuracy = self.split_model.score(X, y)
            if accuracy == 1.0:
                node.is_leaf = True
                node.model = None
                node.result = y[0]
                return node
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if (left_mask.sum() < self.min_samples_leaf or
                        right_mask.sum() < self.min_samples_leaf):
                    continue

                y_left, y_right = y[left_mask], y[right_mask]

                if self.split_policy == 'Gain':
                    gain = gain_ratio(y, y_left, y_right)
                else:
                    gain = self._accuracy_gain(X, y, left_mask, right_mask)

                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_mask': left_mask,
                        'right_mask': right_mask
                    }

        if best_split is None:
            node.is_leaf = True
            node.model = clone(self.base_model).fit(X, y)
            return node

        # Apply best split
        node.split_rule = best_split['feature_index']
        node.split_value = best_split['threshold']

        X_left, y_left = X[best_split['left_mask']], y[best_split['left_mask']]
        X_right, y_right = X[best_split['right_mask']], y[best_split['right_mask']]

        node.left = self._fit_node(X_left, y_left, depth + 1)
        node.right = self._fit_node(X_right, y_right, depth + 1)

        return node
    def fit(self, X, y):
        self.root = self._fit_node(X, y, 0)
        return self
    def _predict_node(self, node, x):
        if node.is_leaf:
            if node.model is None:
                return node.result
            return node.model.predict([x])[0]
        if x[node.split_rule] <= node.split_value:
            return self._predict_node(node.left, x)
        else:
            return self._predict_node(node.right, x)
    def predict(self, X):
        return np.array([self._predict_node(self.root, x) for x in X])
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)



from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
X, y = make_classification(n_samples=200, n_features=5, n_informative=3,
                           n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model = TessellatedLinearModel(base_model=SGDClassifier(), split_policy = 'Accuracy', split_model=LogisticRegression(), max_depth=3, min_samples_leaf=5, min_samples_split=10)
model.fit(X_train, y_train)
print("Accuracy:", model.score(X_test, y_test))


