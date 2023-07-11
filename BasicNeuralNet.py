import os
os.environ["DEBUG"] = "2"
from tinygrad import nn
from tinygrad.tensor import Tensor
from tinygrad.state import safe_save, safe_load, get_state_dict, load_state_dict
import tinygrad.nn.optim as optim
from tinygrad.nn.optim import Adam
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
from numpy import exp, random
import numpy

print("before")

class NuNet():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = Tensor.randn(3,1)

    def _sigmoid(self,x):
        return 1/ (1 + exp(-x))
    
    def _sigmoid_der(self, x):
        return x*(1-x)
    
    def train(self, trn_in, trn_out, trn_num):
        for iteration in range(trn_num):
            output = self.think(trn_in)
            error = trn_out - output
            adjustment = trn_in.T.matmul((error.matmul(self._sigmoid_der(output))))
            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self._sigmoid(inputs.matmul(self.synaptic_weights))



if __name__ == "__main__":
    print("Starting!")
    neural_network = NuNet()

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights.numpy())

    training_set_inputs = Tensor(
        [[0,0,1],
         [1,1,1],
         [1,0,1],
         [0,1,1]]
    )

    training_set_outputs = Tensor(
        [[0],
         [1],
         [1],
         [0]]
    )

    neural_network.train(training_set_inputs, training_set_outputs, 1100)

    print("New synaptic weights: ")
    print(neural_network.synaptic_weights.numpy())

    print("New situation = [1,0,0]")
    newData = Tensor([[1,0,0]])
    print(neural_network.think(newData).numpy())

    newData = Tensor([[1,1,1]])
    print(neural_network.think(newData).numpy())

print("test")