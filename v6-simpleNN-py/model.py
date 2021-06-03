import torch.nn as nn








class model(nn.Module):
    def __init__(self, architecture):
        super(model, self).__init__()

        num_layers = architecture.size
        self.layers = []
        for i in range(num_layers):
            layer = nn.Linear(architecture[i, i+1])
            self.layers.append(layer)

    #forward pass through the net
    def forward(self, input):

        y = input
        for layer in self.layers:
            y = layer(y)

        return y

    def set_params(self, params):
        self.load_state_dict(params, strict=True)

    def get_params(self):
        return self.state_dict
        