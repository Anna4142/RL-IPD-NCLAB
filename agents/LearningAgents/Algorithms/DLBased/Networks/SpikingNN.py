import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
torch.autograd.set_detect_anomaly(True)

class FullyConnectedSNN(nn.Module):
    def __init__(self, architecture, beta=0.9):
        super(FullyConnectedSNN, self).__init__()
        print(architecture)
        self.layers = nn.ModuleList()
        self.spiking_layers = nn.ModuleList()  # Create a module list for spiking layers

        # Initialize layers with corresponding spiking neurons
        for i in range(len(architecture) - 1):
            self.layers.append(nn.Linear(architecture[i], architecture[i+1], bias=True))
            self.spiking_layers.append(snn.Leaky(beta=beta))  # Adding a Leaky spiking neuron for each layer

    def forward(self, x):
        print("Input to network:", x.shape)  # Print input shape
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)  # Apply the linear transformation
            x = self.spiking_layers[i](x)  # Apply spiking neuron dynamics

            # Ensure x is a tensor; assuming spiking layer outputs a tensor directly
            if isinstance(x, tuple):
                x = x[0]  # Assuming the tensor is the first element in the tuple

        x = self.layers[-1](x)  # Output layer does not have spiking dynamics
        return x

    def forward_return_all(self, x):
        all_neurons_output = []
        print("Input to network (all outputs):", x.shape)  # Print initial input shape
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            print(f"Intermediate output before spiking layer {i}:", x.shape)  # Intermediate shapes
            x = self.spiking_layers[i](x)
            all_neurons_output.append(x)
            print(f"State after spiking layer {i}:", x.shape)  # State after spiking layer

        x = self.layers[-1](x)
        all_neurons_output.append(x)
        print("Final output in forward_return_all:", x.shape)  # Final output shape
        return all_neurons_output
