import torch
import torch.nn as nn








class model(nn.Module):
    def __init__(self, architecture):
        super(model, self).__init__()

        num_layers = architecture.size - 1
        self.layers = []
        for i in range(num_layers):
            self.layer = nn.Linear(architecture[i], architecture[i+1])
            self.layers.append(self.layer)

    #forward pass through the net
    def forward(self, input):
        y = input
        for layer in self.layers:
            y = layer(y)

        return y


    def train(self, X_train, y_train, optimizer, criterion):
        #iterate through data
        for i , datapoint in enumerate(X_train):
            # zero the optimizer gradients
            optimizer.zero_grad()

            ### forward pass, backward pass, optimizer step
            out = self.forward(datapoint)
            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()

    def test(self, X_test, y_test, criterion):
        correct = 0
        with torch.no_grad():
            for (x, y) in zip(X_test, y_test):
                output = self.forward(x)
                #loss = criterion(output, y)
                # for now, only look at accuracy, using criterion we can expand this later on 
                _, prediction = torch.max(output.data, 1)
                correct += (prediction == y)
        # return accuracy
        return (correct / X_test.size)

    def set_params(self, params):
        for model_layer, param_layer in zip (self.layers, params):
            model_layer.load_state_dict(param_layer, strict=True)

    def get_params(self):
        parameters = []
        for layer in self.layers:
            parameters.append(layer.state_dict())
        return parameters
        
