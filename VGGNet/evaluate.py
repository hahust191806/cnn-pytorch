import numpy as np 
import os 
import torch 
import torch.nn as nn 

from model import VGG16
from utils import get_test_loader, precision_recall_f1, confusion_matrix


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 10 
    
    # defined model 
    model = torch.load('path_model_ckpt').to(device)
    model.eval()
    # defined dataset 
    train_loader = get_test_loader('./data', batch_size=64)

    all_predict = []
    all_true = []
    # fit 
    for idx, (x_test, label) in enumerate(train_loader):
        # move tensors to the configured device 
        x_test = x_test.to(device)
        label = label.to(device)

        # forward pass 
        y_pred = model(x_test.float()).detach()
        y_pred = torch.argmax(y_pred, dim=-1)
        all_predict.append(y_pred)
        all_true.append(label)
        
    all_predict = torch.cat(all_predict)
    all_true = torch.cat(all_true)
        
    precision, recall, f1 = precision_recall_f1(all_true, all_predict, num_classes)
    for label in range(num_classes):
        print(f"Label {label} có Precision: {precision[label]}, Recall: {recall[label]}, F1: {f1[label]}")
        
    print(confusion_matrix(all_true, all_predict, num_classes=num_classes))