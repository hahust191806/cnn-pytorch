import numpy as np 
import os 
import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss 
from torch.optim import SGD 

from model import AlexNet
from utils import get_train_loader


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # defined hyperparameters 
    num_classes = 10 
    num_epochs = 20 
    batch_size = 64 
    learning_rate = 0.005 
    
    # defined model 
    model = AlexNet(num_classes=10).to(device)

    # loss and optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

    # defined dataset 
    train_loader = get_train_loader('./data', batch_size=batch_size)

    # train 
    for epoch in range(num_epochs):
        for idx, (x_train, label) in enumerate(train_loader):
            # move tensors to the configured device 
            x_train = x_train.to(device)
            label = label.to(device)

            # forward pass 
            y_pred = model(x_train)
            loss = criterion(y_pred, label)

            # backward and optimize 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Iteration: {idx}, Loss: {loss.item()}")
            
        if not os.path.isdir("models"): 
            os.mkdir("models")
        torch.save(model, f"models/mnist_{epoch}.pkl")
    
    print("Model finished training!")