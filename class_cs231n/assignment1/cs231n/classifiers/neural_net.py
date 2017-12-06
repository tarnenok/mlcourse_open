from __future__ import print_function

import numpy as np


class TwoLayerNet(object):
    """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def _relu(self, scores):
        return np.maximum(scores, 0, scores)

    def _softmax(self, scores):
        exp_scores = np.exp(scores)
        exp_sum = np.sum(exp_scores, axis=1)
        return exp_scores/exp_sum.reshape(-1, 1)

    def loss(self, X, y=None, reg=0.0):
        """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        scores1 = X.dot(W1)
        scores1_full = scores1 + b1
        hidden1 = self._relu(scores1_full)
        scores2 = hidden1.dot(W2)
        scores2_full = scores2 + b2

        if y is None:
            return scores2_full

        exp_scores = np.exp(scores2_full - np.sum(scores2_full, axis=1).reshape(-1, 1))
        preds = exp_scores/np.sum(exp_scores, axis=1).reshape(-1, 1)
        data_loss = np.sum(-np.log(preds[np.arange(N), y]))
        data_loss /= N
        reg_loss = reg * np.sum(W1 * W1) + reg * np.sum(W2 * W2)
        loss = reg_loss + data_loss

        grads = {}

        d_scores2_full = preds
        d_scores2_full[np.arange(N), y] -= 1
        d_scores2_full /= N
        grads['b2'] = np.sum(d_scores2_full, axis=0)
        d_scores2 = d_scores2_full
        grads['W2'] = hidden1.T.dot(d_scores2) + W2*2*reg
        d_hidden1 = d_scores2.dot(W2.T)
        d_scores1_full = np.zeros(scores1_full.shape)
        d_scores1_full[scores1_full > 0] = 1
        d_scores1_full *= d_hidden1
        grads['b1'] = np.sum(d_scores1_full, axis=0)
        d_scores1 = d_scores1_full
        grads['W1'] = X.T.dot(d_scores1) + W1*2*reg

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices, :]
            y_batch = y[indices]

            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['b2'] -= learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        z1 = X.dot(W1) + b1
        a1 = np.maximum(0, z1)  # pass through ReLU activation function
        scores = a1.dot(W2) + b2

        exp_scores = np.exp(scores)
        y_pred = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return np.argmax(y_pred, axis=1)

if __name__ == "__main__":
    input_size = 4
    hidden_size = 10
    num_classes = 3
    num_inputs = 5


    def init_toy_model():
        np.random.seed(0)
        return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)


    def init_toy_data():
        np.random.seed(1)
        X = 10 * np.random.randn(num_inputs, input_size)
        y = np.array([0, 1, 2, 2, 1])
        return X, y


    net = init_toy_model()
    X, y = init_toy_data()

    scores = net.loss(X, y, reg=0.05)