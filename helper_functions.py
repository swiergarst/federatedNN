import numpy as np
import torch



def average(in_params, set_sizes, class_imbalances, dataset, use_sizes= False, use_imbalances = False) :

    ### set the parameters dictionary to all zeros before aggregating 
    if dataset == 'banana' :

        parameters= {
        'fc1.weight' : torch.zeros((4,2), dtype=torch.double),
        'fc1.bias' : torch.zeros((4), dtype=torch.double),
        'fc2.weight' : torch.zeros((2,4), dtype=torch.double),
        'fc2.bias' : torch.zeros((2), dtype=torch.double)
    }
    elif dataset == 'MNIST':
        parameters= {
        'fc1.weight' : torch.zeros((100,28*28), dtype=torch.double),
        'fc1.bias' : torch.zeros((100), dtype=torch.double),
        'fc2.weight' : torch.zeros((10,100), dtype=torch.double),
        'fc2.bias' : torch.zeros((10), dtype=torch.double)
    }
    elif dataset == 'MNIST_2class_IID':
        parameters= {
        'fc1.weight' : torch.zeros((100,28*28), dtype=torch.double),
        'fc1.bias' : torch.zeros((100), dtype=torch.double),
        'fc2.weight' : torch.zeros((2,100), dtype=torch.double),
        'fc2.bias' : torch.zeros((2), dtype=torch.double)
    }

    #create size-based weights
    num_clients = set_sizes.size
    weights = np.ones_like(set_sizes)/num_clients

    
    if use_sizes:
        total_size = np.sum(set_sizes) 
        weights = set_sizes / total_size
    
    #do averaging
    for param in parameters.keys():
        for i in range(num_clients):
            parameters[param] += weights[i] * in_params[i][param]

    return parameters