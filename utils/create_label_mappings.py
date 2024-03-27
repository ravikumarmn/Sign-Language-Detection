import os
import json

# Local imports
from src.constants import *


label_to_index = {}
index_to_label = {}
label_index = 0

# Iterate over the labels in the dataset
for label in os.listdir(SAVED_DATA_PATH):
    if label not in label_to_index:
        label_to_index[label] = label_index
        index_to_label[label_index] = label
        label_index += 1

def index_to_word(index):
    return index_to_label.get(index, "Unknown")


