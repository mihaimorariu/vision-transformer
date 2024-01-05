import argparse
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor


def main(args):
    logging.basicConfig(level=logging.DEBUG)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = MNIST(
        root="../datasets",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=8,
    )

    test_dataset = MNIST(
        root="../datasets",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=8,
    )

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    logging.info(f"Using device: {device}")

    if is_cuda:
        logging.info(f"CUDA detected: {torch.cuda.get_device_name(device)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", "-b", type=int, default=128)
    parser.add_argument("--seed", "-s", type=int, default=1234)

    args = parser.parse_args()
    main(args)
