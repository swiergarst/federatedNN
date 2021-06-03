import numpy as np
import torch
from model import model

architecture = np.array([2,4,2])
testModel = model(architecture)

num_layers = architecture.size - 1

parameters = []

# for layer in range(num_layers):
#     param = {
#         'weight': torch.as_tensor(np.zeros((architecture[layer], architecture[layer + 1])).T),
#         'bias' : torch.as_tensor(np.zeros((architecture[layer + 1])))
#     }
#     parameters.append(param)


#print(parameters)

print(testModel.get_params())

param = {
    'fc1.weight' : torch.as_tensor(np.zeros((4,2))),
    'fc1.bias' : torch.as_tensor(np.zeros((4))),
    'fc2.weight' : torch.as_tensor(np.zeros((2,4))),
    'fc2.bias' : torch.as_tensor(np.zeros((2)))
}

testModel.set_params(param)
print(testModel.get_params())