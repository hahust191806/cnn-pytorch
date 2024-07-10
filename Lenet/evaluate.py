"""
    Model evaluation metrics and functions
    
    Metrics for Multi-classification: Accuracy, Precision, Recall, F1-score, Confusion Matrix
""" 
import numpy as np 
import os 
import torch 
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms 
from torchvision.datasets import MNIST
# from torch.utils.tensorboard import SummaryWriter

from utils import precision_recall_f1, confusion_matrix


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load model 
    model = torch.load("models\mnist_9.pkl")
    model.eval()
    # print(model)
    # print(model.state_dict())
    # defined test datasets
    test_dataset = MNIST(root='./test', train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=256)
    all_predict_y = []
    all_label_y = []
    # test dataset 
    for idx, (x_test, label) in enumerate(test_loader):
        x_test = x_test.to(device)
        label = label.to(device)
        predict_y = model(x_test.float()).detach() # separate tensor from computational graph 
        predict_y = torch.argmax(predict_y, dim=-1)
        all_predict_y.append(predict_y)
        all_label_y.append(label)
        
    all_predict_y = torch.cat(all_predict_y)
    all_label_y = torch.cat(all_label_y)
    
    num_classes = 10
    precision, recall, f1 = precision_recall_f1(all_label_y, all_predict_y, num_classes)
    for label in range(num_classes):
        print(f"Label {label} có Precision: {precision[label]}, Recall: {recall[label]}, F1: {f1[label]}")
        
    print(confusion_matrix(all_label_y, all_predict_y, num_classes=num_classes))