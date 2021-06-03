### imports
import torch 
from .model import model
import torch.optim as optim



### master task (will not be used in the current setup, as the researcher file will do this)
def master_task():
    pass




### RPC task
### This will contain the main training loop, including the forward and backward passes
def RPC_train_and_test(data, architecture, weights, criterion, optimizer = 'SGD'):
    ### create net from given architeture
    net = model(architecture)

    X_train, y_train, X_test, y_test = data

    ### initialize the weights from input
    net.set_params(weights)

    ### create optimizer 
    if (optimizer == 'SGD'):
        opt = optim.SGD(net.parameters, lr=5e-1)

    ### train the model
    net.train(X_train, y_train, opt, criterion)

    ### test the model
    test_results = net.test(X_test, y_test)

    ### return the new weights and the test results
    return net.get_params(), test_results