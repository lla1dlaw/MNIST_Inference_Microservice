
# Importing dependencies
import torch
from PIL import Image
from torch import nn,save,load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# Loading Data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="data", download=True, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Define the image classifier model
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 22 * 22, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Create an instance of the image classifier model
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
classifier = ImageClassifier().to(device)

# Define the optimizer and loss function
optimizer = Adam(classifier.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train the model
for epoch in tqdm(range(10)):  # Train for 10 epochs
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Reset gradients
        outputs = classifier(images)  # Forward pass
        loss = loss_fn(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

# Save the trained model
torch.save(classifier.state_dict(), 'model_state.pt')

# Load the saved model
with open('model_state.pt', 'rb') as f: 
     classifier.load_state_dict(load(f))  
       

# # Perform inference on an image
# img = Image.open('image.jpg')
# img_transform = transforms.Compose([transforms.ToTensor()])
# img_tensor = img_transform(img).unsqueeze(0).to(device)
# output = classifier(img_tensor)
# predicted_label = torch.argmax(output)
# print(f"Predicted label: {predicted_label}")
