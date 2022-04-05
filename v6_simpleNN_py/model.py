import torch
import torch.nn as nn
import numpy as np
import math
import sys







class model(nn.Module):
    def __init__(self, dataset, model_choice, ci):
        super(model, self).__init__()
        self.dataset = dataset
        self.ci = ci
        self.model_choice = model_choice
        
        if self.model_choice == "CNN":    
            self.conv_layers, self.lin_layers = self.get_model(dataset)
        else:
            self.lin_layers = self.get_model(dataset)
        

        #params = self.get_params()
        
        #for param in params:
        #    self.ci[param] = torch.zeros_like(params[param])

    def get_model(self, dataset):
        if dataset == "banana":
            return nn.Sequential(
               nn.Linear(2,4),
               nn.Linear(2,4) 
            )
        elif dataset == "MNIST" or dataset == "fashion_MNIST":
            if self.model_choice == "FNN": 
                return nn.Sequential(            
                    nn.Linear(28*28,100),
                    nn.ReLU(),
                    nn.Linear(100,10)
                )
            elif self.model_choice == "CNN" : 
                convLayers = nn.Sequential(
                    nn.Conv2d(1,1, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                linLayers = nn.Sequential(
                    nn.Linear(196, 10)
                    )
                return convLayers, linLayers
        elif dataset == "MNIST_2class":
            if self.model_choice == "FNN":
                return nn.Sequential(
                    nn.Linear(28*28, 100),
                    nn.ReLU(),
                    nn.Linear(100,2) 
                )
            elif self.model_choice == "CNN":
                convLayers = nn.Sequential(
                    nn.Conv2d(1,1, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                linLayers = nn.Sequential(
                    nn.Linear(196, 2)
                    )
                return convLayers, linLayers
        elif dataset == "MNIST_4class" :
            if self.model_choice == "FNN":
                return nn.Sequential(
                    nn.Linear(28*28, 100),
                    nn.ReLU(),
                    nn.Linear(100,4)    
                )
            elif self.model_choice == "CNN":
                convLayers = nn.Sequential(
                    nn.Conv2d(1,1, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                linLayers = nn.Sequential(
                    nn.Linear(196, 4)
                    )
                return convLayers, linLayers
        elif dataset == "A2_PCA" or dataset == "3node" : 
            if self.model_choice == "FNN": 
                return nn.Sequential(
                    nn.Linear(100, 100),
                    nn.ReLU(),
                    nn.Linear(100,2)    
                )
            elif self.model_choice == "CNN":
                convLayers = nn.Sequential(
                    nn.Conv2d(1,1, kernel_size=3, stride=1, padding=1),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                linLayers = nn.Sequential(nn.Linear(25,2))
                return(convLayers, linLayers)
            else :
                raise ValueError("no known model selection supplied")
        else:
            raise ValueError("no known dataset supplied")
    #forward pass through the net
    def forward(self, input):
        #print(input)
        #print(input.shape)

        
        if self.model_choice == "CNN":
            input = self.conv_layers(input)
            input = input.view(input.shape[0], -1)
        return self.lin_layers(input)
  


    def train(self, X_train, y_train, optimizer, criterion, lr, epochs, batch_amount, c,  scaffold, early_stopping, threshold):
    #print(X_train)
    #iterate through data
        batch_size = math.floor(X_train.size()[0]/batch_amount)

        for e in range (epochs):

            for batch in range(batch_amount):
                X_train_batch = X_train[batch* batch_size: (batch+1) * batch_size]
                y_train_batch = y_train[batch* batch_size: (batch+1) * batch_size]
                # zero the optimizer gradients
                optimizer.zero_grad()
                #print(datapoint)
                ### forward pass, backward pass, optimizer step
                out = self.forward(X_train_batch)
                #print(out)
                loss = criterion(out, y_train_batch)
                loss.backward()
                #self.DGD_update(lr, nb_parameters)
                if scaffold :
                    if batch == batch_amount - 1:
                        self.scaffold_update(lr, c, True, batch_amount)
                    else:
                        self.scaffold_update(lr, c, False, batch_amount)
                else : 
                    optimizer.step()

            if early_stopping and loss < threshold:
                break
            return e
            #sys.exit()

    def test(self, X_test, y_test, criterion):
        correct = 0
        with torch.no_grad():
            #for (x, y) in zip(X_test, y_test):
            output = self.forward(X_test)
            #loss = criterion(output, y)
            # for now, only look at accuracy, using criterion we can expand this later on 
            _, prediction = torch.max(output.data, 1)
            correct += (prediction == y_test).sum().item()
        # return accuracy
        return (correct / X_test.size()[0])

    def set_params(self, params):
        self.load_state_dict(params)
        #for model_layer, param_layer in zip (self.layers, params):
        #    model_layer.load_state_dict(param_layer, strict=True)

    def get_params(self):
        return self.state_dict()
        #parameters = []
        #for layer in self.layers:
        #    parameters.append(layer.state_dict())
        #return parameters
        
    def scaffold_update(self, lr, c, use_c, batch):
        params = self.get_params()
        updated_param_dict = {}
        for para, param in zip(self.parameters(), params):
            updated_param_dict[param] = params[param] - lr * (para.grad + c[param] - self.ci[param])
            if use_c:
                self.ci[param] = self.ci[param] - c[param] + ((1/(batch*lr)) * (params[param] - updated_param_dict[param]))

        self.set_params(updated_param_dict)
        #print(lr)
'''
    def DGD_update(self, lr, parameters):
        num_neighbours = parameters.shape[0]
        params = self.get_params()
        updated_param_dict = {}
        for para, param in zip(self.parameters(), params):
            accum_param = torch.zeros_like(para)
            for i in range(num_neighbours):
                accum_param += (1/num_neighbours) * parameters[i][param]
            updated_param_dict[param] = accum_param - lr * para.grad
        self.set_params(updated_param_dict)
'''