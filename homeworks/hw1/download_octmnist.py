#!/usr/bin/env python

import argparse
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import medmnist

"""
This script allows you to download the OCTMNIST dataset to a compressed .npz file, 
which hw1-q2.py and hw1-q3.py can load.
"""


def main():
    """
    Downloads the OCTMNIST dataset to the specified directory.

    Usage:
        python download_octmnist.py --root <root>
    """
    args = parse_args()
    root = args.root

    if not os.path.exists(root):
        os.makedirs(root)
    else:
        assert os.path.isdir(root), f"{root} is not a directory"

    _ = medmnist.OCTMNIST(split="train", download=True, root=root)


def parse_args():
    """
    Parses command-line arguments.

    :return: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    return parser.parse_args()


if __name__ == "__main__":
    main()
