### imports
import torch 
from .model import model
import torch.optim as optim
from sklearn.model_selection import train_test_split


### master task (will not be used in the current setup, as the researcher file will do this)
def master_task():
    pass




### RPC task
### This will contain the main training loop, including the forward and backward passes
def RPC_train_and_test(data, architecture, parameters, criterion, optimizer = 'SGD'):
    ### create net from given architeture
    net = model(architecture)

    #load in the data file in the correct format
    dim_num = 2
    dims = ['dim' + str(i) for i in range(dim_num)]
    x_tot = data[dims].values
    y_tot = data['label'].values
    X_train, X_test, y_train, y_test = train_test_split(x_tot, y_tot, test_size = 0.20, random_state=42)

    ### initialize the weights and biases from input
    net.set_params(parameters)

    ### create optimizer 
    if (optimizer == 'SGD'):
        opt = optim.SGD(net.parameters, lr=5e-1)

    ### train the model
    net.train(X_train, y_train, opt, criterion)

    ### test the model
    test_results = net.test(X_test, y_test)

    ### return the new weights and the test results
    return net.get_params(), test_results