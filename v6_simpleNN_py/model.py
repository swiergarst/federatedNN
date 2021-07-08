import torch
import torch.nn as nn
import numpy as np
import sys







class model(nn.Module):
    def __init__(self, dataset, model_choice):
        super(model, self).__init__()
        self.dataset = dataset
        self.ci = {}
        self.model_choice = model_choice
        if self.model_choice == "CNN":    
            self.conv_layers, self.lin_layers = self.get_model(dataset)
        else:
            self.lin_layers = self.get_model(dataset)

        params = self.get_params()

        for param in params:
            self.ci[param] = torch.zeros_like(params[param])

    def get_model(self, dataset):
        if dataset == "banana":
            return nn.Sequential(
               nn.Linear(2,4),
               nn.Linear(2,4) 
            )
        elif dataset == "MNIST":
            return nn.Sequential(            
                nn.Linear(28*28,100),
                nn.ReLU(),
                nn.Linear(100,10)
            )
        elif dataset == "MNIST_2class_IID":
            if model == "FNN":
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


    def train(self, X_train, y_train, optimizer, criterion, lr, c,  scaffold, use_c):
    #print(X_train)
    #iterate through data
        # zero the optimizer gradients
        optimizer.zero_grad()
        #print(datapoint)
        ### forward pass, backward pass, optimizer step
        out = self.forward(X_train)
        #print(out)
        loss = criterion(out, y_train)
        loss.backward()
        if scaffold :
            self.scaffold_update(lr, c, use_c)
        else : 
            optimizer.step()
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
        
    def scaffold_update(self, lr, c, use_c):
        params = self.get_params()
        updated_param_dict = {}
        for para, param in zip(self.parameters(), params):
            updated_param_dict[param] = params[param] - lr * (para.grad + c[param] - self.ci[param])
            if use_c:
                self.ci[param] = self.ci[param] - c[param] + (1/lr) * (params[param] - updated_param_dict[param])

        self.set_params(updated_param_dict)
        #print(lr)
