# TODO add jit
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from mnist_classifier_app.models.mnist_classifier_model import MNISTClassifierModel


def create_tensorboard_writer(log_directory: str):
    return SummaryWriter(log_dir=log_directory)


def get_model(model_path=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    get_logger().info(f"Selected device: {device}")
    model = MNISTClassifierModel().to(device)
    model.load_state_dict(torch.load(model_path)) if model_path is not None else None
    return model


def get_devices(num_devices=4):
    if torch.cuda.is_available():
        device_count = min(num_devices, torch.cuda.device_count())
        devices = [f"cuda:{i}" for i in range(device_count)]
    else:
        devices = ["cpu"]
    return devices


def create_logger(**kwargs):
    if "log_path" not in kwargs:
        # path
        # os.makedirs(".."+ os.sep + ".."+ os.sep + "application_log" + os.sep + "mnist_classifier_app.log", exist_ok=True)
        kwargs["log_path"] = "application_log" + os.sep + "mnist_classifier_app.log"
    if "verbose" in kwargs:
        # level
        kwargs["log_level"] = logging.DEBUG
    else:
        # level
        kwargs["log_level"] = logging.ERROR
    logger = logging.getLogger("mnist_classifier_app")
    # set level
    logger.setLevel(kwargs["log_level"])
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # add a rotating handler
    file_handler = RotatingFileHandler(kwargs["log_path"], maxBytes=20 * (1024**2), backupCount=1)  # 20MB
    file_handler.setFormatter(formatter)

    # add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # add handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def get_logger():
    return logging.getLogger("mnist_classifier_app")


create_logger(verbose=True)


def read_yaml(path: str):
    with open(path) as config_file:
        try:
            return yaml.safe_load(config_file)
        except yaml.YAMLError as exc:
            get_logger().error("Read yaml error: " + str(exc))
    return None
