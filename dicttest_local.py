import torch


def master_task():
    pass


def RPC_dicttest(data, setting):

    # this dict doesn't work
    if setting == 1:
        parameters= {
                'lin_layers.0.weight' : torch.randn((100,28*28), dtype=torch.double),
                'lin_layers.0.bias' : torch.randn((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.randn((2,100), dtype=torch.double),
                'lin_layers.2.bias' : torch.randn((2), dtype=torch.double)
        }                

    # this one does
    elif setting == 2:
        parameters = {
                    'conv_layers.0.weight': torch.randn((1,1,3,3), dtype=torch.double),
                    'conv_layers.0.bias' : torch.randn(1, dtype=torch.double),
                    'lin_layers.0.weight' : torch.randn((2, 196), dtype=torch.double),
                    'lin_layers.0.bias' : torch.randn(2, dtype=torch.double)
                }

    return (parameters)
        