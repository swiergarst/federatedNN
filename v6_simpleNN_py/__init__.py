### imports
from numpy import double
import torch 
from .model import model
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

### master task (will not be used in the current setup, as the researcher file will do this)
def master_task():
    pass




### RPC task
### This will contain the main training loop, including the forward and backward passes
def RPC_train_and_test(data, parameters, criterion, model_choice, lr = 5e-1, local_epochs = 1, local_batch_amt = 1, scaffold = False, c = None,  optimizer = 'SGD', dataset = 'banana', use_c = True):
    ### create net from given architeture
    net = model(dataset, model_choice)
    net = net.double() #apparently I need this

    #load in the data file in the correct format
    if dataset == 'banana':
        dim_num = 2
        dims = ['dim' + str(i) for i in range(dim_num)]
        x_tot = data[dims].values
        y_tot = data['label'].values
        X_train_arr, X_test_arr, y_train_arr, y_test_arr = train_test_split(x_tot, y_tot, test_size = 0.20, random_state=42)
    
        X_train = torch.as_tensor(X_train_arr, dtype=torch.double)
        X_test = torch.as_tensor(X_test_arr, dtype=torch.double)
        y_train = torch.as_tensor(y_train_arr, dtype=torch.int64)
        y_test = torch.as_tensor(y_test_arr, dtype=torch.int64)
    elif dataset == 'MNIST' or dataset == 'MNIST_2class' or dataset == "MNIST_4class":
        dim_num = 784
        dims = ['pixel' + str(i) for i in range(dim_num)]
        X_train_arr = data.loc[data['test/train'] == 'train'][dims].values
        y_train_arr = data.loc[data['test/train'] == 'train']['label'].values
        X_test_arr = data.loc[data['test/train'] == 'test'][dims].values
        y_test_arr = data.loc[data['test/train'] == 'test']['label'].values
        X_train = torch.as_tensor(X_train_arr, dtype=torch.double)
        X_test = torch.as_tensor(X_test_arr, dtype=torch.double)
        y_train = torch.as_tensor(y_train_arr, dtype=torch.int64)
        y_test = torch.as_tensor(y_test_arr, dtype=torch.int64)

        if model_choice == "CNN":
            X_train = X_train.reshape(X_train.shape[0], 1, 28,28)
            X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
        num_samples = X_train.size()[0]
    ### initialize the weights and biases from input
    net.set_params(parameters)

    ### create optimizer 
    if (optimizer == 'SGD'):
        opt = optim.SGD(net.parameters(), lr=lr)
        
   ### test the model
   # we test before training such that the same model is used as in the server
    test_results = net.test(X_test, y_test, criterion)

    ### train the model
    net.train(X_train, y_train, opt, criterion, lr, local_epochs, local_batch_amt, c, scaffold, use_c)

 
    ### return the new weights and the test results
    return [net.get_params(), test_results, num_samples, net.ci]