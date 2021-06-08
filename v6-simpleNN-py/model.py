import torch
import torch.nn as nn








class model(nn.Module):
    def __init__(self, architecture):
        super(model, self).__init__()

        #num_layers = architecture.size - 1
        ## the 'old' architecture
        #self.fc1 = nn.Linear(2,4)
        #self.fc2 = nn.Linear(4,2)

        self.fc1 = nn.Linear(28*28,100)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100,10)

    #forward pass through the net
    def forward(self, input):
        #print(input)
        y1 = self.fc1(input)
        y1 = self.relu1(y1)
        #y = input
        #for layer in self.layers:
        #    y = layer(y)

        return self.fc2(y1)


    def train(self, X_train, y_train, optimizer, criterion):
    #print(X_train)
    #iterate through data
    #for x, y in zip(X_train, y_train):
        # zero the optimizer gradients
        #print(y)
        #print(X_train)
        #y = nn.functional.one_hot(y_train, num_classes=2)
        #print(y)
        optimizer.zero_grad()
        #print(datapoint)
        ### forward pass, backward pass, optimizer step
        out = self.forward(X_train)
        #print(out)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

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
        
