import numpy as np 
import os 
import torch 
from torch.nn import CrossEntropyLoss 
from torch.optim import SGD 
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms 
from torchvision.datasets import MNIST
# from torch.utils.tensorboard import SummaryWriter

from model import LeNet


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Tạo một SummaryWriter để lưu trữ log cho TensorBoard
    # writer = SummaryWriter()
    batch_size = 256 
    # defined datasets
    train_dataset = MNIST(root='./train', train=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    # defined model 
    model = LeNet()
    sgd = SGD(model.parameters(), lr=1e-1) # defined Optimizer 
    loss_fn = CrossEntropyLoss() # defined loss function 
    epochs = 10
    for epoch in range(epochs):
        model.train() # convert model to train mode 
        for idx, (x_train, label) in enumerate(train_loader):
            x_train = x_train.to(device)
            label = label.to(device)
            sgd.zero_grad() # set gradient value of weights to zero 
            predict_y = model(x_train.float())
            loss = loss_fn(predict_y, label.long()) # calculate loss
            loss.backward() # calculate gradient
            sgd.step() # update weights 
            
            # Ghi lại log cho TensorBoard
            # writer.add_scalar('Loss/train', loss.item(), epoch)
            
            print(f"Epoch {epoch+1}, Iteration: {idx}, Loss: {loss.item()}")
            
        if not os.path.isdir("models"): 
            os.mkdir("models")
        torch.save(model, f"models/mnist_{epoch}.pkl")
        
    # Đóng SummaryWriter
    # writer.close()
        
    print("Model finished training!")