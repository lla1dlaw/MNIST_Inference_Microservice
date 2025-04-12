#Feed Forward Neural Network Trained and Tested on MNIST dataset
import pprint
import os
from matplotlib.pylab import f
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Predictor import NeuralNet
from tqdm import tqdm
from tqdm import trange


# devicce config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using GPU") if torch.cuda.is_available() else print('Using CPU')

def train_model(model: NeuralNet, num_epochs: int, train_loader, criterion, optimizer):
    accuracies = []
    losses = []

    data_length = len(train_loader.dataset)
    acc = "-"
    lss = "-"

    bar = trange( # progress bar for training
        num_epochs, 
        desc="Training", 
        unit="epoch", 
        leave=True, 
        total=num_epochs,
        dynamic_ncols=True,
        colour="green",
        postfix= {"Acc": acc, "Loss": lss},
        bar_format="{l_bar}{bar}| {postfix}",
        unit_scale=True
        )
    
    bar_step = 1/data_length # update progress bar every 1% of data length
    
    #training loop
    for epoch in bar:
        correct_predictions = 0
        loss_accumulator = 0
        for i, (images, labels) in enumerate(train_loader):
            # (100, 1, 28, 28)
            # (100, 784)
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            #forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # accumulate acuracy and loss
            loss_accumulator += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            
            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            bar.update(bar_step) # update progress bar

        epoch_accuracy = correct_predictions / data_length
        epoch_loss = loss_accumulator / data_length
        accuracies.append(epoch_accuracy) 
        losses.append(epoch_loss)
        # update progress bar with new epoch accuracy and loss values
        bar.set_postfix({"Acc": epoch_accuracy, "Loss": epoch_loss})
    bar.close()
    print(f"\nFinal Loss: {epoch_loss}, Final Accuracy: {epoch_accuracy:.5f}")

    return accuracies, losses

        
def test_model(model, test_loader):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')


def save_csv(model, path):
        state = model.state_dict()

        os.makedirs(path, exist_ok=True)
        for key, value in state.items():
            pd.DataFrame(value.cpu().numpy()).to_csv(f"{path}\\{key}.csv", index=False, header=False)
        
        print(f"Model saved to: {os.getcwd()}\\{path}")


def print_and_save(model, dimensions):

    if input("Would you like to save this model? (y/n): ").lower() == "y":
        if input("Save as .pt or .csv? (pt/csv): ").lower() == "pt":
            torch.save(model, f"model{dimensions}.pt")
            print(f"Model saved to: {os.getcwd()}\\params.pt")
        else:
            save_csv(model, f"model{dimensions}.pt")


def load_data(batch_size: int):
    # Load MNIST
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def plot_graphs(accuracies: list[float], losses: list[float], num_epochs: int):

    fig, axis_1 = plt.subplots()
    axis_1.set_xlabel('Epochs')
    axis_1.set_ylabel('Accuracy')
    x_axis = np.arange(1, num_epochs+1)
    axis_1.plot(x_axis, accuracies, label='Accuracy', color='tab:blue')
    axis_1.tick_params(axis='y', labelcolor='tab:blue')

    axis_2 = axis_1.twinx()
    axis_2.set_ylabel('Loss')
    axis_2.plot(x_axis, losses, label='Loss', color='tab:orange')
    axis_2.tick_params(axis='y', labelcolor='tab:orange')

    fig.tight_layout()
    plt.title('Training Accuracy and Loss')
    plt.legend()
    plt.show()


def main():
    # hyperparameters
    input_size = 784  # 28x28
    num_classes = 10
    hidden_widths = [100, 50]
    num_epochs = 20
    batch_size = 100
    learning_rate = 0.001

    train_loader, test_loader = load_data(batch_size)

    model = NeuralNet(input_size, hidden_widths, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    print("\nModel Loaded")
    accuracies, losses = train_model(model, num_epochs, train_loader, criterion, optimizer)
    
    # display training accuracy and loss
    plot_graphs(accuracies, losses, num_epochs)

    test_model(model, test_loader)
    print_and_save(model, hidden_widths)


if __name__ == "__main__":
    main()