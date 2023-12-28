import os
import random

import numpy as np
import torch


def configure_seed(seed):
    """
    Configures the seed for the random, numpy, and torch libraries.

    :param seed: the seed to use
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_oct_data(bias=False, eq_test_dist=False):
    """
    Loads the preprocessed, featurized octmnist dataset, optionally adding a bias feature.

    :param bias: whether to add a bias feature
    :param eq_test_dist: whether to equalize the test distribution
    :return: a dictionary containing the training, development, and test sets
    """
    data = np.load('octmnist.npz')
    train_X = data["train_images"].reshape([data["train_images"].shape[0], -1]) / 256
    dev_X = data["val_images"].reshape([data["val_images"].shape[0], -1]) / 256
    test_X = data["test_images"].reshape([data["test_images"].shape[0], -1]) / 256

    train_y = np.asarray(data["train_labels"]).squeeze()
    val_y = np.asarray(data["val_labels"]).squeeze()
    test_y = np.asarray(data["test_labels"]).squeeze()
    if not eq_test_dist:
        test_y_class0 = test_y[test_y == 0][0:182]  # 182
        test_X_class0 = test_X[test_y == 0][0:182]  # 182
        test_y_class1 = test_y[test_y == 1][0:55]  # 55
        test_X_class1 = test_X[test_y == 1][0:55]  # 55
        test_y_class2 = test_y[test_y == 2][0:42]  # 42
        test_X_class2 = test_X[test_y == 2][0:42]  # 42
        test_y_class3 = test_y[test_y == 3][0:250]  # 250
        test_X_class3 = test_X[test_y == 3][0:250]  # 250
        test_X = np.vstack((test_X_class0,
                            test_X_class1,
                            test_X_class2,
                            test_X_class3))
        test_y = np.hstack((test_y_class0,
                            test_y_class1,
                            test_y_class2,
                            test_y_class3))
    if bias:
        train_X = np.hstack((train_X, np.ones((train_X.shape[0], 1))))
        dev_X = np.hstack((dev_X, np.ones((dev_X.shape[0], 1))))
        test_X = np.hstack((test_X, np.ones((test_X.shape[0], 1))))
    return {"train": (train_X, np.asarray(data["train_labels"]).squeeze()),
            "dev": (dev_X, np.asarray(data["val_labels"]).squeeze()),
            "test": (test_X, test_y)}  # np.asarray(data["test_labels"]).squeeze())}


class ClassificationDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for the octmnist dataset.
    """

    def __init__(self, data):
        """
        Initializes the dataset.

        :param data: the dict returned by utils.load_oct_data
        """
        train_X, train_y = data["train"]
        dev_X, dev_y = data["dev"]
        test_X, test_y = data["test"]

        self.X = torch.tensor(train_X.reshape(train_X.shape[0], 1, 28, 28), dtype=torch.float32)
        self.y = torch.tensor(train_y, dtype=torch.long)

        self.dev_X = torch.tensor(dev_X.reshape(dev_X.shape[0], 1, 28, 28), dtype=torch.float32)
        self.dev_y = torch.tensor(dev_y, dtype=torch.long)

        self.test_X = torch.tensor(test_X.reshape(test_X.shape[0], 1, 28, 28), dtype=torch.float32)
        self.test_y = torch.tensor(test_y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
