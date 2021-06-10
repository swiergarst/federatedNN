# with lots of inspiration from the courselabs from the CS4240 deep learning course and fedML

### imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))


from vantage6.tools.mock_client import ClientMockProtocol
from v6_simpleNN_py.model import model

# parameters
criterion = nn.CrossEntropyLoss()
optimizer = 'SGD'

num_global_rounds = 20
num_clients = 10
dataset = 'MNIST'

if dataset == 'banana':
    datasets =  ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/banana/banana_dataset_client" + str(i) + ".csv" for i in range(10)]
elif dataset == 'MNIST':
    datasets= ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST/MNIST_dataset_client" + str(i) + ".csv" for i in range(10)]

### connect to server
client = ClientMockProtocol(
    #datasets= ["/home/swier/Documents/afstuderen/nnTest/v6-simpleNN-py/local/banana/banana_dataset_client" + str(i) + ".csv" for i in range(10)],
    
    datasets= datasets,
    module="v6_simpleNN_py"
)

organizations = client.get_organizations_in_my_collaboration()
org_ids = [organization["id"] for organization in organizations]




### parameter settings

#architecture array will have the following format:
# [layer_0_input_dim, layer_1_input_dim, .. , layer_n_input_dim, layer_n_output_dim]
# so if the array is of length 2 there will be a single layer, 3 params equals 2 layers with the second param the size of the second layer, etc
# this is not counting the output as a layer    

architecture = np.array([2,4,2])

#dataset = 'MNIST'

torch.manual_seed(42)
#create the weights and biases
# moons parameters
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
    round_task = client.create_new_task(
        input_= {
            'method' : 'train_and_test',
            'kwargs' : {
                'parameters' : parameters,
                'criterion': criterion,
                'optimizer': optimizer,
                'dataset' : dataset
            }
        },
        organization_ids=org_ids
    )

    ### aggregate responses
    results = np.array(client.get_results(round_task.get("id")))
    #print(results[:,1])
    local_parameters = results[:,0]
    acc_results[:, round] = results[:,1]

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
    ### generate new model parameters
    for param in parameters.keys():
        for i in range(num_clients):
            parameters[param] += local_parameters[i][param]
        parameters[param] /= num_clients


# finally, do one test on 'all' test data
MNIST_test = torch.load("/home/swier/Documents/afstuderen/MNIST/processed/test.pt")
X_test = MNIST_test[0].flatten(start_dim=1)/255
y_test = MNIST_test[1]
testModel = model(dataset)
testModel.set_params(parameters)
complete_test_results  = testModel.test(X_test, y_test, criterion)

#print(acc_results)
print(complete_test_results)
print(np.mean(acc_results, axis=0))
x = np.arange(num_global_rounds)
plt.plot(x, np.mean(acc_results, axis=0))
plt.plot(num_global_rounds,complete_test_results)
plt.show()
