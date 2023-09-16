from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms


class CustomMNISTDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        label = row["label"]
        image = row.drop("label").values.astype(float)

        # Reshape image to 2D
        image = np.reshape(image, (28, 28))

        if self.transform is not None:
            image = self.transform(image)

        return image, label
