import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

#Local imports
from src.constants import *
from utils.create_label_mappings import label_to_index 

class KeypointsDataset(Dataset):
    def __init__(self, data_path, label_to_index):
        self.data_path = data_path
        self.samples = []
        self.label_to_index = label_to_index
        self.index_to_label = {index: label for label, index in label_to_index.items()}

        for label, label_index in label_to_index.items():
            label_path = os.path.join(data_path, label)
            for video_folder in os.listdir(label_path):
                video_folder_path = os.path.join(label_path, video_folder)
                if os.path.isdir(video_folder_path):
                    for file in os.listdir(video_folder_path):
                        if file.endswith('.npy'):
                            keypoints = np.load(os.path.join(video_folder_path, file))
                            label_tensor = torch.tensor(label_index, dtype=torch.long)
                            self.samples.append((keypoints, label_tensor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        keypoints, label = self.samples[idx]
        return torch.tensor(keypoints, dtype=torch.float32), label
    
   
# Create dataset instance
dataset = KeypointsDataset(SAVED_DATA_PATH, label_to_index)

# Split dataset into train and test
train_size = int(TRAIN_SPLIT_SIZE * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create data loaders for train and test sets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

