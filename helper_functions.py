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



def get_datasets(dataset, class_imbalance = False, sample_imbalance = False):
    if dataset == 'banana':
        datasets =  ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/banana/banana_dataset_client" + str(i) + ".csv" for i in range(10)]
    elif dataset == 'MNIST':
        datasets= ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST/MNIST_dataset_client" + str(i) + ".csv" for i in range(10)]
    elif dataset == 'MNIST_2class_IID':
        if class_imbalance:
            datasets = ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_class_imbalance/MNIST_2Class_class_imbalance_client" + str(i) + ".csv" for i in range(10)]
        elif sample_imbalance:
            datasets =["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_Sample_Imbalance/MNIST_2Class_sample_imbalance_client" + str(i) + ".csv" for i in range(10)]
        else:
            datasets= ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_IID/MNIST_2Class_IID_client" + str(i) + ".csv" for i in range(10)]
    return datasets