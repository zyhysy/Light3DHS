import argparse


class BaseOptim(object):
    """Base class for optimizer"""

    def __init__(self) -> None:
        self.eps = 1e-8
        self.lr = 0.001
        self.weight_decay = 1e-4

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser
