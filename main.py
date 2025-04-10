

#Feed Forward Neural Network Trained and Tested on MNIST dataset
import pprint
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import os

# devicce config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using GPU") if torch.cuda.is_available() else print('Using CPU')


class NeuralNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_hidden: int, num_classes: int):
        super(NeuralNet, self).__init__()

        self.input_size = input_size
        
        self.layers = [nn.Linear(input_size, hidden_size)]

        for i in range(num_hidden-2): 
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        self.layers.append(nn.Linear(hidden_size, num_classes))
        self.layers = nn.ModuleList(self.layers)
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        # no softmax at the end
        return x
    

def train_model(model: NeuralNet, num_epochs: int, train_loader, criterion, optimizer):
    accuracies = []
    losses = []

    data_length = len(train_loader.dataset)
    #training loop
    n_total_step = len(train_loader)
    for epoch in range(num_epochs):
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

        epoch_accuracy = correct_predictions / data_length
        epoch_loss = loss_accumulator / data_length
        
        if epoch+1 % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}, Accuracy: {epoch_accuracy:.5f}")

        accuracies.append(epoch_accuracy) 
        losses.append(epoch_loss)

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


def print_and_save(model):
    if input("See state dictionary of the model? (y/n): ").lower() == "y":
        print(f"\nState Dictionary Type: {type(model.state_dict())}")
        pprint.pp(model.state_dict())

    if input("Would you like to save this model? (y/n): ").lower() == "y":
        save_csv(model, "params")


def load_data(batch_size: int):
    # Load MNIST
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def main():
    # hyperparameters
    input_size = 784  # 28x28
    num_classes = 10
    hidden_size = 100
    num_hidden = 4
    num_epochs = 20
    batch_size = 100
    learning_rate = 0.001

    # load data
    train_loader, test_loader = load_data(batch_size)

    # initialize model
    model = NeuralNet(input_size, hidden_size, num_hidden, num_classes).to(device)

    #Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # train model
    accuracies, loss = train_model(model, num_epochs, train_loader, criterion, optimizer)

    plt.plot(accuracies, label='Accuracy')
    plt.plot(loss, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Training Accuracy and Loss')
    plt.show()

    test_model(model, test_loader)
    print_and_save(model)


if __name__ == "__main__":
    main()