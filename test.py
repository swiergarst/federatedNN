from typing import Sized
import numpy as np
from helper_functions import init_params, heatmap
import torch

num_clients = 3
num_rounds = 5

global_data = {
    "a" : torch.arange(start=0, end=100) * 0.5
}




client1_data = {
    "a" : torch.arange(start=0,end=100)* 0.75
}

client2_data = {
    "a" : torch.zeros(100)
}

client3_data = {
    "a" : torch.arange(start=0, end=100)* 0.5
}
#print(client3_data)
clients = [client1_data, client2_data, client3_data]

testmap = heatmap(num_clients, num_rounds)

for round in range(num_rounds):
    client3_data["a"] *= 2
    testmap.save_round(round, clients, global_data)

testmap.show_map()