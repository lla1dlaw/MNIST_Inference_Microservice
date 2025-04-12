import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size: int, hidden_widths: list[int], num_classes: int):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        previous_width = hidden_widths[0]
        # create list of layers
        self.layers = [nn.Linear(input_size, previous_width)]
        for width in hidden_widths[1:-1]: 
            self.layers.append(nn.Linear(previous_width, width))
            previous_width = width
        self.layers.append(nn.Linear(previous_width, num_classes))
        self.layers = nn.ModuleList(self.layers)
        self.activation = nn.ReLU()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        # no softmax at the end
        return x
    
