import numpy as np


def svm_loss_naive(W, X, y, reg):
    """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
    dW = np.zeros(W.shape)

    delta = 1
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + delta
            if margin > 0:
                dW[:, j] += X[i]
                dW[:, y[i]] += - X[i]
                loss += margin

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W: np.ndarray, X: np.ndarray, y: np.ndarray, reg: float):
    """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
    delta = 1
    scores = X.dot(W)
    margins = scores - scores[np.arange(scores.shape[0]), y].reshape(-1, 1)
    margins += delta
    margins[np.arange(margins.shape[0]), y] = 0
    margins[margins <= 0] = 0

    loss = np.sum(margins)
    loss /= len(X)
    loss += reg * np.sum(W * W)

    X_mask = np.zeros(margins.shape)
    X_mask[margins > 0] = 1
    incorrect_counts = np.sum(X_mask, axis=1)
    X_mask[np.arange(X.shape[0]), y] = -incorrect_counts
    dW = X.T.dot(X_mask)
    dW /= X.shape[0]
    dW += reg * W

    return loss, dW
