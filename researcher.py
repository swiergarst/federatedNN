# with lots of inspiration from the courselabs from the CS4240 deep learning course and fedML

### imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from vantage6.tools.mock_client import ClientMockProtocol

### connect to server
client = ClientMockProtocol(
    datasets= ["/home/swier/Documents/afstuderen/nnTest/v6-simpleNN-py/local/banana/banana_dataset_client" + str(i) + ".csv" for i in range(10)],
    module="v6-simpleNN-py"
)

organizations = client.get_organizations_in_my_collaboration()
org_ids = [organization["id"] for organization in organizations]]




### parameter settings

#architecture array will have the following format:
# [layer_0_input_dim, layer_1_input_dim, .. , layer_n_input_dim, layer_n_output_dim]
# so if the array is of length 2 there will be a single layer, 3 params equals 2 layers with the second param the size of the second layer, etc
# this is not counting the output as a layer    

architecture = np.array([2,4,2])
criterion = nn.CrossEntropyLoss()
optimizer = 'SGD'

global_rounds = 1
num_clients = 10
dataset = 'banana'

#create the weights and biases; this requires some acrobatics b/c of the specific format
parameters = []

for layer in range(architecture.size - 1):
    param = {
        'weight': torch.as_tensor(np.zeros((architecture[layer], architecture[layer + 1])).T),
        'bias' : torch.as_tensor(np.zeros((architecture[layer + 1])))
    }
    parameters.append(param)

### main loop
for round in range(global_rounds):

    ### request task from clients
    round_task = client.create_new_task(
        input_= {
            'method' : 'train_and_test',
            'kwargs' : {
                'architecture' : architecture,
                'parameters' : parameters,
                'criterion': criterion,
                'optimizer': optimizer
            }
        },
        organization_ids=org_ids
    )

    ### aggregate responses
    results = client.get_results(round_task.get("id"))
    print(results)
    ### generate new model parameters