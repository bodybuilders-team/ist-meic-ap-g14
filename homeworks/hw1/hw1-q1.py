#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import matplotlib.pyplot as plt
import numpy as np

import utils


class LinearModel(object):
    """
    Base class for linear models.
    """

    def __init__(self, n_classes, n_features):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        """
        Update the weight vector on a single training example.

        :param x_i: training example
        :param y_i: gold label
        :param kwargs: other arguments are ignored
        """
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        """
        Train the model for one epoch.

        :param X: training examples
        :param y: gold labels
        :param kwargs: other arguments are ignored
        """
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """
        Predict labels for examples.

        :param X: examples to predict labels for (n_examples x n_features)
        :return: predicted labels (n_examples)
        """
        scores = self.W.dot(X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        Evaluate the accuracy of the model on a dataset.

        :param X: examples to evaluate the model on (n_examples x n_features)
        :param y: gold labels
        :return: accuracy of the model on the provided examples (float)
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    """
    Perceptron model.
    """

    def update_weight(self, x_i, y_i, **kwargs):
        """
        Update the weight vector on a single training example.

        :param x_i: (n_features) a single training example
        :param y_i: (scalar) the gold label for that example
        :param kwargs: other arguments are ignored
        """
        # Q1.1a
        scores = self.W.dot(x_i.T)  # vector of scores for each class
        y_hat = scores.argmax()  # choose the class with the highest score
        if y_hat != y_i:
            self.W[y_i, :] += x_i  # update the weight vector for the correct class
            self.W[y_hat, :] -= x_i


class LogisticRegression(LinearModel):
    """
    Logistic regression model.
    """

    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        Update the weight vector on a single training example, using the stochastic gradient descent update rule:

        W_{t+1} = W_t - learning_rate * gradient, where gradient = (y_i - softmax(W_t * x_i)) * x_i

        :param x_i: (n_features) a single training example
        :param y_i: the gold label for that example
        :param learning_rate: (float) keep it at the default value for your plots
        """
        # Q1.1b
        label_scores = np.expand_dims(self.W.dot(x_i.T), axis=1) # (n_classes x 1)

        # One-hot encoding of the gold label
        y_one_hot = np.zeros(label_scores.shape)
        y_one_hot[y_i] = 1

        # Softmax
        probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))

        # Gradient
        gradient = (y_one_hot - probabilities) * x_i

        # Update the weights: W_{t+1} = W_t - learning_rate * gradient
        self.W += learning_rate * gradient


class MLP(object):
    """
    Multi-layer perceptron model.
    """

    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer
        self.W1 = np.random.normal(0.1, 0.1, (hidden_size, n_features))  # input to hidden layer
        self.b1 = np.zeros(hidden_size)  # biases of hidden layer

        self.W2 = np.random.normal(0.1, 0.1, (n_classes, hidden_size))  # hidden to output layer
        self.b2 = np.zeros(n_classes)  # biases of output layer

    def predict(self, X):
        """
        Predict labels for examples.

        :param X: examples to predict labels for (n_examples x n_features)
        :return: predicted labels (n_examples)
        """
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        predicted_labels = []
        for x_i in X:
            h0 = x_i
            z1 = self.W1.dot(h0) + self.b1
            h1 = np.maximum(z1, 0)  # ReLU

            z2 = self.W2.dot(h1) + self.b2

            # Softmax
            probabilities = softmax(z2)
            predicted_labels.append(probabilities.argmax())

        return predicted_labels

    def evaluate(self, X, y):
        """
        Evaluate the accuracy of the model on a dataset.

        :param X: examples to evaluate the model on (n_examples x n_features)
        :param y: gold labels
        :return: accuracy of the model on the provided examples (float)
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Train the model for one epoch.

        :param X: training examples
        :param y: gold labels
        :param learning_rate: (float) keep it at the default value for your plots
        :return: loss of the epoch (float).
        """
        total_loss = 0

        for x_i, y_i in zip(X, y):
            # Compute the forward pass
            h0 = x_i
            z1 = self.W1.dot(h0) + self.b1
            h1 = np.maximum(z1, 0)  # ReLU

            z2 = self.W2.dot(h1) + self.b2

            # Compute the loss - cross-entropy loss
            # One-hot encoding of the gold label
            y = np.zeros(z2.shape)
            y[y_i] = 1

            probabilities = softmax(z2)
            loss = -y.dot(np.log(probabilities + 1e-8))  # add a small constant to avoid log(0)
            total_loss += loss

            # Compute the backward pass
            # Gradient of the loss w.r.t. the output layer
            grad_z2 = probabilities - y

            # Gradient of the loss w.r.t. the hidden layer parameters
            grad_W2 = grad_z2[:, None].dot(h1[:, None].T)
            grad_b2 = grad_z2

            # Gradient of the loss w.r.t. the hidden layer activations
            grad_h1 = self.W2.T.dot(grad_z2)

            # Gradient of the loss w.r.t. the hidden layer
            grad_z1 = grad_h1 * (z1 > 0)  # ReLU

            # Gradient of the loss w.r.t. the hidden layer parameters
            grad_W1 = grad_z1[:, None].dot(h0[:, None].T)
            grad_b1 = grad_z1


            # Update the weights
            self.W1 -= learning_rate * grad_W1
            self.b1 -= learning_rate * grad_b1
            self.W2 -= learning_rate * grad_W2
            self.b2 -= learning_rate * grad_b2

        return total_loss / X.shape[0]


def plot(epochs, train_accs, val_accs, filename=None):
    """
    Plot the accuracy curves.

    :param epochs: array of epoch numbers
    :param train_accs: array of training accuracies
    :param val_accs: array of validation accuracies
    """
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.savefig(filename)
    plt.show()


def plot_loss(epochs, loss, filename=None):
    """
    Plot the loss curve.

    :param epochs: array of epoch numbers
    :param loss: array of training losses
    """
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.savefig(filename)
    plt.show()


def softmax(x):
    """
    Compute the softmax of vector x.
    :param x: vector of scores
    :return: result of softmax
    """
    x = x - np.max(x)  # standardize the scores to avoid overflow
    return np.exp(x) / np.sum(np.exp(x))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []

    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )

        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
    ))

    # plot
    plot(epochs, train_accs, valid_accs,
         filename=f'images/{opt.model}_accs_{opt.epochs}epochs{"_" + str(opt.learning_rate) + "lr" if opt.model != "perceptron" else ""}.png')
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f'images/{opt.model}_loss_{opt.epochs}.png')


if __name__ == '__main__':
    main()
