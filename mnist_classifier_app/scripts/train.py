from __future__ import annotations

import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import mnist_classifier_app.utils as utils
from mnist_classifier_app.datasets.custom_dataset import CustomMNISTDataset
from mnist_classifier_app.losses.cross_entropy_loss import CustomCrossEntropyLoss


# TODO add tensorboard enable option


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    tensorboard_writer,
    learning_rate=0.001,
    save_check_point=1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in tqdm(range(1, num_epochs + 1), colour="magenta"):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device).float()
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step = epoch * len(train_loader) + i
            tensorboard_writer.add_scalar("Loss/train", loss.item(), step)  # noqa

        utils.get_logger().info(f"Epoch [{epoch}/{num_epochs}] - Loss: {running_loss / len(train_loader)}")
        if save_check_point != 0 and epoch % save_check_point == 0:
            os.makedirs("saved_mnist_classifier_models", exist_ok=True)
            torch.save(model.state_dict(), "saved_mnist_classifier_models" + os.sep + f"{str(epoch)}_mnist_classifier.pth")
            utils.get_logger().info(f"Epoch_{str(epoch)}_mnist_classifier.pth has been saved.")
        for name, param in model.named_parameters():
            tensorboard_writer.add_histogram(name + "_weights", param.data, epoch)
            tensorboard_writer.add_histogram(name + "_gradients", param.grad, epoch)
        validation(model=model, loader=val_loader, tensorboard_writer=tensorboard_writer, device=device) if val_loader is not None else None

    tensorboard_writer.close()


def validation(model, loader, tensorboard_writer=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, position=0, leave=True, colour="cyan"):
            images = images.to(device).float()
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        validation_accuracy = 100 * correct / total
    utils.get_logger().info(f"Accuracy: {validation_accuracy}")
    tensorboard_writer.add_scalar("Accuracy/validation", validation_accuracy, 0) if tensorboard_writer is not None else None  # noqa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path of th config file.", default=f"configuration_files{os.sep}configuration_file.yml")
    args = parser.parse_args()
    config_file = args.config

    train_config = utils.read_yaml(config_file)["train"]

    # Parse Config
    log_dir = "tensorboard_logs" + os.sep + "simple"  # TODO fix name
    train_dataset_path = train_config["train_dataset_path"]
    validation_dataset_path = train_config["validation_dataset_path"] if "validation_dataset_path" in train_config else None
    batch_size = train_config["batch_size"]
    learning_rate = train_config["learning_rate"]
    num_epochs = train_config["number_of_epochs"]
    model_save_path = train_config["model_save_path"]
    save_check_point = train_config["save_check_point"]
    device = train_config["device"]
    model_path = train_config["model_path"] if "model_path" in train_config else None

    criterion = CustomCrossEntropyLoss()
    dataset = CustomMNISTDataset(csv_file=train_dataset_path)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) if validation_dataset_path is not None else None
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    tensorboard_writer = utils.create_tensorboard_writer(log_directory=log_dir)
    model = utils.get_model(device=device, **{"model_path": model_path} if model_path else {})
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        learning_rate=learning_rate,
        save_check_point=save_check_point,
        tensorboard_writer=tensorboard_writer,
        device=device,
    )
