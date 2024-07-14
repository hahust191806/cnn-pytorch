from torch.utils.data import Dataset 
import os 
import cv2


idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}
"""
    The structure of the dataset class is something like this: 
    Class CustomDataset(Dataset):
        1. __init__() Function 
        2. __len__() Function : which just returns the length of the dataset.
        3. __getitem__() Function : This processes and returns 1 datapoint at a time.
"""
class CustomDataset(Dataset):
    def __init__(self, dataset_path: list = None, transform: bool = False):
        self.dataset_path = dataset_path
        self.transform = transform
        
    def __len__(self):
        pass 
    
    def __getitem__(self, idx):
        image_filepath = self.dataset_path[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = image_filepath.split('/')[-2]
        label = class_to_idx[label]
        if self.transform is not None: 
            image = self.transform(image=image)["image"]

        return image, label 