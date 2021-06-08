# with lots of inspiration from the courselabs from the CS4240 deep learning course and fedML

### imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time

from io import BytesIO
from vantage6.tools.util import info
from vantage6.client import Client
### connect to server

print("Attempt login to Vantage6 API")
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "1234")

client.setup_encryption(None)


#organizations = client.get_organizations_in_my_collaboration()
#org_ids = [organization["id"] for organization in organizations]




### parameter settings

#architecture array will have the following format:
# [layer_0_input_dim, layer_1_input_dim, .. , layer_n_input_dim, layer_n_output_dim]
# so if the array is of length 2 there will be a single layer, 3 params equals 2 layers with the second param the size of the second layer, etc
# this is not counting the output as a layer    

architecture = np.array([2,4,2])
criterion = nn.CrossEntropyLoss()
optimizer = 'SGD'

ids = [i for i in range(1,11)]

num_global_rounds = 20
num_clients = 10
dataset = 'banana'

torch.manual_seed(42)
#create the weights and biases
if dataset == 'banana':
    parameters= {
        'fc1.weight' : torch.randn((4,2), dtype=torch.double),
        'fc1.bias' : torch.randn((4), dtype=torch.double),
        'fc2.weight' : torch.randn((2,4), dtype=torch.double),
        'fc2.bias' : torch.randn((2), dtype=torch.double)
    }
elif dataset == 'MNIST' : 
# mnist parameters
    parameters= {
        'fc1.weight' : torch.randn((100,28*28), dtype=torch.double),
        'fc1.bias' : torch.randn((100), dtype=torch.double),
        'fc2.weight' : torch.randn((10,100), dtype=torch.double),
        'fc2.bias' : torch.randn((10), dtype=torch.double)
    }


acc_results = np.zeros((num_clients, num_global_rounds))

### main loop
for round in range(num_global_rounds):

    ### request task from clients
    round_task = client.post_task(
        input_= {
            'method' : 'train_and_test',
            'kwargs' : {
                'architecture' : architecture,
                'parameters' : parameters,
                'criterion': criterion,
                'optimizer': optimizer
            }
        },
        name = "nntest round " + str(round),
        image = "sgarst/federated-learning:nnTest",
        organization_ids=ids,
        collaboration_id= 1
    )
    info("Waiting for results")
    res = client.get_results(task_id=round_task.get("id"))
    attempts=1
    
    while(None in [res[i]["result"] for i in range(num_clients)]  and attempts < 7):
        print("waiting...")
        time.sleep(5)
        res = client.get_results(task_id=round_task.get("id"))
        attempts += 1

    info("Obtaining results")
    #result  = client.get_results(task_id=task.get("id"))
    result = []
    for i in range(num_clients):
        result.append(np.load(BytesIO(res[i]["result"]),allow_pickle=True))
    

    results = np.array(result)
    #print(np.array(results[0,1]))
    #print(results[:,1])
    local_parameters = np.array(results[:,0])
    acc_results[:, round] = np.array(results[:,1])

    ### set the parameters dictionary to all zeros before aggregating
    if dataset == 'banana' :

        parameters= {
        'fc1.weight' : torch.zeros((4,2), dtype=torch.double),
        'fc1.bias' : torch.zeros((4), dtype=torch.double),
        'fc2.weight' : torch.zeros((2,4), dtype=torch.double),
        'fc2.bias' : torch.zeros((2), dtype=torch.double)
    }
    elif dataset == 'MNIST':
        parameters= {
        'fc1.weight' : torch.zeros((100,28*28), dtype=torch.double),
        'fc1.bias' : torch.zeros((100), dtype=torch.double),
        'fc2.weight' : torch.zeros((10,100), dtype=torch.double),
        'fc2.bias' : torch.zeros((10), dtype=torch.double)
    }
    for param in parameters.keys():
        for i in range(num_clients):
            parameters[param] += local_parameters[i][param]
        parameters[param] /= num_clients
print(acc_results)
print(np.mean(acc_results, axis=0))
x = np.arange(num_global_rounds)
plt.plot(x, np.mean(acc_results, axis=0))
plt.show()
    ### generate new model parameters
