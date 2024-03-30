from __future__ import annotations

import unittest

import torch

from mnist_classifier_app.models.mnist_classifier_model import MNISTClassifierModel


class ModelTest(unittest.TestCase):
    classification_model = MNISTClassifierModel()
    dummy_tensor = torch.randn(32, 1, 28, 28)

    def test_is_model_valid(self):
        output = self.classification_model(self.dummy_tensor)
        print(output.shape)
