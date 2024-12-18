"""
Module for specifying the input argument parsers of the training script entry points.
"""
import argparse
from typing import List
import logging


def parse_input_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-d",
        "--data_path",
        help="Directory containing train and validate folders with feature files",
        type=str,
        required=True,
        metavar="<dir>",
        action="store",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Directory to place output from training",
        type=str,
        metavar="<dir>",
        action="store",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--pretrained_model",
        required=False,
        help="path to pretrained model (often named training_model.pth)",
        action="store",
        metavar="<path>",
        type=str,
        default="",
    )
    args = parser.parse_args()
    return args
