### imports
from numpy import double
import torch 
from .model import model
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import math

### master task (will not be used in the current setup, as the researcher file will do this)
def master_task():
    pass




### RPC task
### This will contain the main training loop, including the forward and backward passes
def RPC_train_and_test(data, parameters, model_choice, nb_parameters = None, dgd = False, criterion = torch.nn.CrossEntropyLoss(),  lr = 5e-1, local_epochs = 1, local_batch_amt = 1, scaffold = False, c = None, ci = None,  optimizer = 'SGD', dataset = 'MNIST_2class', early_stopping = False, threshold = 10):
    ### create net from given architeture
    net = model(dataset, model_choice, ci)
    net = net.double() #apparently I need this

    #load in the data file in the correct format
    X_train_arr = data.loc[data['test/train'] == 'train'].drop(columns = ['test/train', 'label']).values
    y_train_arr = data.loc[data['test/train'] == 'train']['label'].values
    X_test_arr = data.loc[data['test/train'] == 'test'].drop(columns = ["test/train", "label"]).values
    y_test_arr = data.loc[data['test/train'] == 'test']['label'].values
    X_train = torch.as_tensor(X_train_arr, dtype=torch.double)
    X_test = torch.as_tensor(X_test_arr, dtype=torch.double)
    y_train = torch.as_tensor(y_train_arr, dtype=torch.int64)
    y_test = torch.as_tensor(y_test_arr, dtype=torch.int64)
    num_samples = X_train.size()[0]
    ### initialize the weights and biases from input
    if dgd:
        params_to_set = {}
        num_neighbours = nb_parameters.shape[0]

        for para in parameters:
            params_to_set[para] = torch.zeros_like(parameters[para])
            for neighbour in range(num_neighbours):
                params_to_set[para] += (1/num_neighbours) * nb_parameters[neighbour][para]
        net.set_params(params_to_set)
    else:
        net.set_params(parameters)

    if model_choice == "CNN":     
        if dataset == "kinase_KDR" or dataset == "kinase_ABL1":
            X_test = X_test.reshape(X_test.shape[0], 1, 4096, 2)
            X_train = X_train.reshape(X_train.shape[0], 1, 4096, 2)
        else:
            reshape_size = int(math.sqrt(X_test.shape[1]))
            X_test = X_test.reshape(X_test.shape[0], 1, reshape_size, reshape_size)
            X_train = X_train.reshape(X_train.shape[0], 1, reshape_size, reshape_size)
                
    ### create optimizer 
    if (optimizer == 'SGD'):
        opt = optim.SGD(net.parameters(), lr=lr)
        
   ### test the model
   # we test before training such that the same model is used as in the server
    test_results = net.test(X_test, y_test, criterion)

    ### train the model
    lepochs_used = net.train(X_train, y_train, opt, criterion, lr, local_epochs, local_batch_amt, c, scaffold, early_stopping, threshold)

 
    ### return the new weights and the test results
    return (net.get_params(), test_results, num_samples, None, lepochs_used)


def RPC_dicttest(data, setting):

    # the FNN return
    if setting == 1:
        parameters= {
                'lin_layers.0.weight' : torch.randn((100,28*28), dtype=torch.double),
                'lin_layers.0.bias' : torch.randn((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.randn((2,100), dtype=torch.double),
                'lin_layers.2.bias' : torch.randn((2), dtype=torch.double)
        }                

    # the CNN return
    elif setting == 2:
        parameters = {
                    'conv_layers.0.weight': torch.randn((1,1,3,3), dtype=torch.double),
                    'conv_layers.0.bias' : torch.randn(1, dtype=torch.double),
                    'lin_layers.0.weight' : torch.randn((2, 196), dtype=torch.double),
                    'lin_layers.0.bias' : torch.randn(2, dtype=torch.double)
                }

    return (parameters)
        
