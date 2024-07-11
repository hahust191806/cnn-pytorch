import numpy as np 
import torch 
import torch.nn as nn 
from torchvision import datasets 
from torchvision import transforms 
import os 

from utils import get_train_loader
from model import VGG16


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # defined hyperparameter 
    num_classes = 10 
    num_epochs = 20
    batch_size = 64 
    learning_rate = 0.005
    
    model = VGG16(num_classes=num_classes)
    
    # get dataset 
    train_dataloader = get_train_loader('dataset', batch_size=batch_size)

    # Loss and Optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

    # training 
    for epoch in range(num_epochs):
        for idx, (x_train, label) in enumerate(train_dataloader):
            # Move tensors to the configured device 
            x_train = x_train.to(device)
            label = label.to(device)

            # Forward pass 
            y_predict = model(x_train)
            loss = criterion(y_predict, label)

            # Backward pass
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Iteration: {idx}, Loss: {loss.item()}")
            
        if not os.path.isdir("models"): 
            os.mkdir("models")
        torch.save(model, f"models/mnist_{epoch}.pkl")
    
    print("Model finished training!")