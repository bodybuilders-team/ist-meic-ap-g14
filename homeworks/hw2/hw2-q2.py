#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

import utils


class CNN(nn.Module):
    """
    Convolutional Neural Network for classification.
    """

    def __init__(self, dropout_prob, no_maxpool=False):
        """
        Initialize CNN.

        :param dropout_prob: the dropout probability
        :param no_maxpool: if True, do not use max pooling
        """
        super(CNN, self).__init__()
        self.no_maxpool = no_maxpool
        if not no_maxpool:
            # Implementation for Q2.1
            raise NotImplementedError
        else:
            # Implementation for Q2.2
            raise NotImplementedError

        # Implementation for Q2.1 and Q2.2
        raise NotImplementedError

    def forward(self, x):
        """
        Forward pass of the CNN.

        :param x: input of shape [b, c, w, h]
        :return: output of shape [b, n_classes]
        """
        # input should be of shape [b, c, w, h]
        # conv and relu layers

        # max-pool layer if using it
        if not self.no_maxpool:
            raise NotImplementedError

        # conv and relu layers

        # max-pool layer if using it
        if not self.no_maxpool:
            raise NotImplementedError

        # prep for fully connected layer + relu

        # drop out
        x = self.drop(x)

        # second fully connected layer + relu

        # last fully connected layer
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


def train_batch(X, y, model, optimizer, criterion, **kwargs):
    """
        Train the model on a single batch of examples.

        :param X: input tensor of shape (batch_size x n_features)
        :param y: label tensor of shape (batch_size)
        :param model: a PyTorch defined model
        :param optimizer: optimizer used in gradient step
        :param criterion: loss function

        :return: loss (float)
    """
    optimizer.zero_grad()
    out = model(X, **kwargs)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X):
    """
    Make predictions on a batch of examples.

    :param model: a PyTorch defined model
    :param X: input tensor of shape (batch_size x n_features)
    :return: predicted labels (tensor of integers of shape (batch_size))
    """
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    Evaluate the model on a dataset.

    :param model: a PyTorch defined model
    :param X: input tensor of shape (n_examples x n_features)
    :param y: label tensor of shape (n_examples)

    :return: accuracy (float)
    """
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    """
    Plot the plottable over the epochs.

    :param epochs: list of epochs
    :param plottable: list of values to plot
    :param ylabel: label for y-axis
    :param name: name of the plot
    """
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


def get_number_trainable_params(model):
    """
    Get the number of trainable parameters in the model.

    :param model: a PyTorch defined model
    :return: number of trainable parameters (int)
    """
    ## TO IMPLEMENT - REPLACE return 0
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.7)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-no_maxpool', action='store_true')

    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_oct_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    # initialize the model
    model = CNN(opt.dropout, no_maxpool=opt.no_maxpool)

    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )

    # get a loss criterion
    criterion = nn.NLLLoss()

    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    # plot
    config = "{}-{}-{}-{}-{}".format(opt.learning_rate, opt.dropout, opt.l2_decay, opt.optimizer, opt.no_maxpool)

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))

    print('Number of trainable parameters: ', get_number_trainable_params(model))


if __name__ == '__main__':
    main()
