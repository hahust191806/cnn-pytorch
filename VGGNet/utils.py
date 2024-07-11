import numpy as np 
import torch 
import torch.nn as nn 
from torchvision import datasets 
from torchvision import transforms 
from torch import Tensor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
"""
    Precision = TP / (TP + FN)
    Recall = TP / (TP + FP)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
"""

def precision_recall_f1(y_true: Tensor = None, y_predict: Tensor = None, num_classes: int = None) -> list[float, float, float]:
    true_positives = torch.zeros(num_classes)
    false_positives = torch.zeros(num_classes)
    false_negatives = torch.zeros(num_classes)

    for idx in range(num_classes):
        true_positives[idx] = torch.sum((y_true == idx) & (y_predict == idx)).item()
        false_positives[idx] = torch.sum((y_true == idx) & (y_predict != idx)).item()
        false_negatives[idx] = torch.sum((y_true != idx) & (y_predict == idx)).item()

    precision = true_positives / (false_negatives + true_positives)
    recall = true_positives / (true_positives + false_positives)
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1 

def confusion_matrix(y_true: Tensor = None, y_predict: Tensor = None, num_class: int = None) -> Tensor: 
    matrix = torch.zeros((num_class, num_class), dtype=torch.int64)

    for t, p in zip(y_true, y_predict):
        matrix[t.long(), p.long()] += 1 
        
    return matrix


"""
    Loading dataset 
"""
def get_train_loader(datadir, batch_size):
    # Normalization 
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    # Composes several transforms 
    train_transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize, 
    ])
    # load the dataset 
    train_dataset = datasets.CIFAR10(
        root=datadir, train=True, 
        download=True, transform=train_transforms, 
    )
    # load the data loader 
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size
    )
    
    return train_loader

def get_test_loader(datadir, batch_size):
    # Normalization 
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    # Compose several transforms 
    test_transforms = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(), 
        normalize,
    ])
    # load the dataset 
    test_dataset = datasets.CIFAR10(
        root=datadir, train=False, 
        download=True, transform=test_transforms,
    )
    # load the data loader 
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size
    )
    
    return test_loader