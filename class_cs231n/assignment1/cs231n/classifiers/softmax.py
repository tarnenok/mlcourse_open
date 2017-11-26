import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
  Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        scores = X[i].dot(W)
        exp_scores = np.exp(scores)
        loss += -np.log(exp_scores[y[i]]/np.sum(exp_scores))
        for j in range(num_classes):
            dW[:, j] += 1/np.sum(exp_scores)*exp_scores[j]*X[i]
            if j == y[i]:
                dW[:, j] -= X[i]

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
    num_train = X.shape[0]

    exp_scores = np.exp(X.dot(W))
    sum_exp_scores = np.sum(exp_scores, axis=1)
    loss = -np.sum(np.log(exp_scores[np.arange(num_train), y]/sum_exp_scores))
    loss /= num_train
    loss += reg * np.sum(W * W)

    soft_max_values = exp_scores/sum_exp_scores.reshape(-1, 1)
    soft_max_values[np.arange(num_train), y] -= 1
    dW = X.T.dot(soft_max_values)

    dW /= num_train
    dW += reg * W

    return loss, dW
