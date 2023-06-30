from __future__ import annotations

import unittest

import torch

from model import MNISTClassifierModel


class ModelTest(unittest.TestCase):
    classification_model = MNISTClassifierModel()
    dummy_tensor = torch.randn(32, 1, 28, 28)

    def test_is_model_valid(self):
        output = self.classification_model(self.dummy_tensor)
        self.assertTrue(output.shape == torch.Size([32, 10]) , msg="Output is not expected. Please check model.")
