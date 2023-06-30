from __future__ import annotations

import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from custom_dataset import CustomMNISTDataset


class CustomDatasetTest(unittest.TestCase):
    test_csv_path = "data" + os.sep + "test_mnist_data.csv"
    custom_dataset = CustomMNISTDataset(test_csv_path)
    data_length = len(custom_dataset)

    def test_length(self):
        expected_length = 10
        self.assertEqual(self.data_length, expected_length, msg="Length of dataset must be 10")

    def test_transforms(self):
        def check_types(expected_image: torch.Tensor, expected_label: np.int64):
            self.assertIsInstance(expected_image, torch.Tensor, msg=f"Data must be a tensor but received {type(expected_image)}")
            self.assertIsInstance(expected_label, np.int64, msg=f"Label must be an integer but received {type(expected_label)}")

        [check_types(self.custom_dataset[index][0], self.custom_dataset[index][1]) for index in range(self.data_length)]

        test_dataframe = pd.read_csv(self.test_csv_path)

        for index in range(self.data_length):
            row = test_dataframe.iloc[index]
            normalized_tensor = self.custom_dataset[index][0]

            transform = transforms.ToTensor()
            image_from_csv = row.drop("label").values.astype(float)
            assertion_tensor = transform(np.reshape(image_from_csv, (28, 28)))

            denormalized_tensor = normalized_tensor * 0.5 + 0.5
            are_equal = torch.eq(assertion_tensor, denormalized_tensor).all()
            self.assertTrue(
                are_equal, msg="Normalization is not correctly done! Normalization must be done with transforms.Normalize((0.5,), (0.5,))."
            )

    def test_get_item(self):
        assertion_list = [1, 0, 1, 4, 0, 0, 7, 3, 5, 3]
        incoming_labels = [self.custom_dataset[index][1] for index in range(self.data_length)]
        self.assertListEqual(assertion_list, incoming_labels, msg="Labels must be came in correct order of assertion list")

    def test_visulize_labels_and_tensors(self):
        def visualize(tensor: torch.tensor, label: np.int64):
            image = tensor.numpy()
            sample = np.squeeze(image)
            sample = (sample + 1) / 2
            plt.imshow(sample, cmap="gray")
            plt.title(f"Label: {label}")
            plt.show()

        [visualize(tensor=self.custom_dataset[index][0], label=self.custom_dataset[index][1]) for index in range(self.data_length)]
