import unittest
from model import MNISTClassifierModel
import torch

class ModelTest(unittest.TestCase):


    classification_model = MNISTClassifierModel()
    dummy_tensor = torch.randn(32, 1, 28, 28)
    def test_is_model_valid(self):
        output = self.classification_model(self.dummy_tensor)
        print(output.shape)

