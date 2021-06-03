# with lots of inspiration from the courselabs from the CS4240 deep learning course and fedML

### imports
import numpy as np
import torch.nn as nn
import torch.optim as optim

### connect to server






### parameter settings

#architecture array will have the following format:
# [layer_0_input_dim, layer_1_input_dim, .. , layer_n_input_dim, layer_n_output_dim]
# so if the array is of length 2 there will be a single layer, 3 params equals 2 layers with the second param the size of the second layer, etc
# this is not counting the output as a layer    

architecture = np.array([2,4,2])
criterion = nn.CrossEntropyLoss()
optimizer = 'SGD'

global_rounds = 1
num_clients = 10
dataset = 'banana'

### main loop


    ### request task from clients


    ### aggregate responses

    ### generate new model parameters