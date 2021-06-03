import numpy as np
import torch
from model import model

architecture = np.array([2,4,2])
testModel = model(architecture)

num_layers = architecture.size - 1

parameters = []

for layer in range(num_layers):
    param = {
        'weight': torch.as_tensor(np.zeros((architecture[layer], architecture[layer + 1])).T),
        'bias' : torch.as_tensor(np.zeros((architecture[layer + 1])))
    }
    parameters.append(param)


#print(parameters)

print(testModel.get_params())

testModel.set_params(parameters)
print(testModel.get_params())