import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
import json


class MfccData(Dataset):
    def __init__(self, mfccs, labels):
        self.mfccs = mfccs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.flatten(torch.Tensor(self.mfccs[idx])), self.labels[idx]


if __name__ == "__main__":
    model = nn.Sequential(
        nn.LSTM(13, 128, batch_first=True),
        nn.Linear(128, 10),
        nn.Softmax(),
    )

    with open("data.json", "r") as f:
        data = json.load(f)
        mfccs = data["mfcc"]
        labels = data["labels"]
    full_dataset = MfccData(mfccs, labels)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    batches = DataLoader(train_dataset, batch_size=32, shuffle=True)
