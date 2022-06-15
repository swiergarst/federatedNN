import torch

import runner_base
import numpy as np
from io import BytesIO

class nn_runner(runner_base):
    def __init__(self):
        super(nn_runner, self).__init__()

    def set_seed(self, run):
        seed = self.seed_offset + run
        torch.manual_seed(seed)
        return seed

    def create_model(self):
        return None

    def params_to_numpy(self, parameters):
        for key in parameters.keys():
                parameters[key] = parameters[key].numpy()
        return (parameters)

    def init_global_params(self, zeros):
        ### set the parameters dictionary to all zeros before aggregating 
        if zeros:
            if self.dataset == 'banana' :
                parameters= {
                'lin_layers.0.weight' : torch.zeros((4,2), dtype=torch.double),
                'lin_layers.0.bias' : torch.zeros((4), dtype=torch.double),
                'lin_layers.1.weight' : torch.zeros((2,4), dtype=torch.double),
                'lin_layers.1.bias' : torch.zeros((2), dtype=torch.double)
            }
            elif self.dataset == 'MNIST' or self.dataset == "fashion_MNIST":
                if self.model_choice == "FNN" :
                    parameters= {
                    'lin_layers.0.weight' : torch.zeros((100,28*28), dtype=torch.double),
                    'lin_layers.0.bias' : torch.zeros((100), dtype=torch.double),
                    'lin_layers.2.weight' : torch.zeros((10,100), dtype=torch.double),
                    'lin_layers.2.bias' : torch.zeros((10), dtype=torch.double)
                }
                elif self.model_choice == "CNN" :
                    parameters = {
                        'conv_layers.0.weight': torch.zeros((1,1,3,3)),
                        'conv_layers.0.bias' : torch.zeros(1),
                        'lin_layers.0.weight' : torch.zeros((10, 196)),
                        'lin_layers.0.bias' : torch.zeros(10)
                    }
            elif self.dataset == 'MNIST_2class':
                if self.model_choice == "FNN":
                    parameters= {
                    'lin_layers.0.weight' : torch.zeros((100,28*28), dtype=torch.double),
                    'lin_layers.0.bias' : torch.zeros((100), dtype=torch.double),
                    'lin_layers.2.weight' : torch.zeros((2,100), dtype=torch.double),
                    'lin_layers.2.bias' : torch.zeros((2), dtype=torch.double)
                    }
                elif self.model_choice == "CNN":
                    parameters = {
                        'conv_layers.0.weight': torch.zeros((1,1,3,3)),
                        'conv_layers.0.bias' : torch.zeros(1),
                        'lin_layers.0.weight' : torch.zeros((2, 196)),
                        'lin_layers.0.bias' : torch.zeros(2)
                    }
            elif self.dataset == "MNIST_4class":
                if self.model_choice == "FNN":
                    parameters= {
                    'lin_layers.0.weight' : torch.zeros((100,28*28), dtype=torch.double),
                    'lin_layers.0.bias' : torch.zeros((100), dtype=torch.double),
                    'lin_layers.2.weight' : torch.zeros((4,100), dtype=torch.double),
                    'lin_layers.2.bias' : torch.zeros((4), dtype=torch.double)
                    }
                elif self.model_choice == "CNN":
                    parameters = {
                        'conv_layers.0.weight': torch.zeros((1,1,3,3)),
                        'conv_layers.0.bias' : torch.zeros(1),
                        'lin_layers.0.weight' : torch.zeros((4, 196)),
                        'lin_layers.0.bias' : torch.zeros(4)
                    }
            elif self.dataset in ["A2_PCA", "3node", "2node"] :
                if self.model_choice == "FNN": 
                    parameters = {
                    'lin_layers.0.weight' : torch.zeros((100, 100), dtype=torch.double),
                    'lin_layers.0.bias' : torch.zeros((100), dtype=torch.double),
                    'lin_layers.2.weight' : torch.zeros((2,100), dtype=torch.double),
                    'lin_layers.2.bias' : torch.zeros((2), dtype=torch.double)
                    }
                elif self.model_choice == "CNN":
                    parameters = {
                        'conv_layers.0.weight': torch.zeros((1,1,3,3)),
                        'conv_layers.0.bias' : torch.zeros(1),
                        'lin_layers.0.weight' : torch.zeros((2, 25)),
                        'lin_layers.0.bias' : torch.zeros(2)
                    }
                else:
                    raise ValueError("model selection not known")
            else:
                raise ValueError("dataset unknown: ", self.dataset)
        else:
            if self.dataset == 'banana':
                parameters= {
                    'lin_layers.0.weight' : torch.randn((4,2), dtype=torch.double),
                    'lin_layers.0.bias' : torch.randn((4), dtype=torch.double),
                    'lin_layers.2.weight' : torch.randn((2,4), dtype=torch.double),
                    'lin_layers.2.bias' : torch.randn((2), dtype=torch.double)
                }
            elif self.dataset == 'MNIST' or self.dataset == "fashion_MNIST": 
            # mnist parameters
                if self.model_choice == "FNN" :
                    parameters= {
                        'lin_layers.0.weight' : torch.randn((100,28*28), dtype=torch.double),
                        'lin_layers.0.bias' : torch.randn((100), dtype=torch.double),
                        'lin_layers.2.weight' : torch.randn((10,100), dtype=torch.double),
                        'lin_layers.2.bias' : torch.randn((10), dtype=torch.double)
                    } 
                elif self.model_choice == "CNN" :
                    parameters = {
                        'conv_layers.0.weight': torch.randn((1,1,3,3)),
                        'conv_layers.0.bias' : torch.randn(1),
                        'lin_layers.0.weight' : torch.randn((10, 196)),
                        'lin_layers.0.bias' : torch.randn(10)
                    }
            elif self.dataset == 'MNIST_2class':
                if self.model_choice == "FNN":
                    parameters= {
                    'lin_layers.0.weight' : torch.randn((100,28*28), dtype=torch.double),
                    'lin_layers.0.bias' : torch.randn((100), dtype=torch.double),
                    'lin_layers.2.weight' : torch.randn((2,100), dtype=torch.double),
                    'lin_layers.2.bias' : torch.randn((2), dtype=torch.double)
                    }
                elif self.model_choice == "CNN":
                    parameters = {
                        'conv_layers.0.weight': torch.randn((1,1,3,3)),
                        'conv_layers.0.bias' : torch.randn(1),
                        'lin_layers.0.weight' : torch.randn((2, 196)),
                        'lin_layers.0.bias' : torch.randn(2)
                    }
            elif self.dataset == 'MNIST_4class':
                if self.model_choice == "FNN":
                    parameters= {
                    'lin_layers.0.weight' : torch.randn((100,28*28), dtype=torch.double),
                    'lin_layers.0.bias' : torch.randn((100), dtype=torch.double),
                    'lin_layers.2.weight' : torch.randn((4,100), dtype=torch.double),
                    'lin_layers.2.bias' : torch.randn((4), dtype=torch.double)
                    }
                elif self.model_choice == "CNN":
                    parameters = {
                        'conv_layers.0.weight': torch.randn((1,1,3,3)),
                        'conv_layers.0.bias' : torch.randn(1),
                        'lin_layers.0.weight' : torch.randn((4, 196)),
                        'lin_layers.0.bias' : torch.randn(4)
                    }
            elif self.dataset in ["A2_PCA", "3node", "2node"]:
                if self.model_choice == "FNN": 
                    parameters = {
                    'lin_layers.0.weight' : torch.randn((100, 100), dtype=torch.double),
                    'lin_layers.0.bias' : torch.randn((100), dtype=torch.double),
                    'lin_layers.2.weight' : torch.randn((2,100), dtype=torch.double),
                    'lin_layers.2.bias' : torch.randn((2), dtype=torch.double)
                    }
                elif self.model_choice == "CNN":
                    parameters = {
                        'conv_layers.0.weight': torch.randn((1,1,3,3)),
                        'conv_layers.0.bias' : torch.randn(1),
                        'lin_layers.0.weight' : torch.randn((2, 25)),
                        'lin_layers.0.bias' : torch.randn(2)
                    }
                else:
                    raise ValueError("model selection not known")
            else:
                raise ValueError("dataset unknown: ", self.dataset)
        return parameters

    def send_task(self, ids):
        # create dict to send to clients
        dict_tosend = self.client_vars.copy()
        
        round_task = self.client.post_task(
            input_ = {
                'method' : 'train_and_test',
                'kwargs' : dict_tosend
            },
            name = "task :)",
            image = self.image,
            organization_ids = ids, 
            collaboration_id = 1
        )
        return round_task

    def send_task_scaffold(self):
        pass

    def get_task_results(self, round_task):
        while None in [res[i]["result"] for i in range(self.num_clients)]:
            res = np.array(self.client.get_results(task_id = round_task.get("id")))

        result = np.empty(self.num_clients, dtype=object)
        final_results = np.empty((3, self.num_clients))
        for i in range(self.num_clients):
            result[i] = np.load(BytesIO(res[i]["result"]) ,allow_pickle=True)
            for j in range(3):
                final_results[j,i] = result[i][j]
        return final_results
        