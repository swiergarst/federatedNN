# with lots of inspiration from the courselabs from the CS4240 deep learning course and fedML

### imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from vantage6.tools.mock_client import ClientMockProtocol
from v6_simpleNN_py.model import model
from helper_functions import average, get_datasets, get_config

start_time = time.time()

# parameters

#torch 
torch.manual_seed(1)
criterion = nn.CrossEntropyLoss()
optimizer = 'SGD'


#dataset
dataset = 'MNIST_2class_IID'
class_imbalance = False
sample_imbalance = False

datasets, parameters, X_test, y_test = get_config(dataset,class_imbalance, sample_imbalance)

#federated settings
num_global_rounds = 5
num_clients = 10

#test model for global testing
testModel = model(dataset)
testModel.double()

# arrays to store results
acc_results = np.zeros((num_clients, num_global_rounds))
global_acc_results = np.zeros((num_global_rounds))
array_save = False

### connect to server
client = ClientMockProtocol(
    datasets= datasets,
    module="v6_simpleNN_py"
)
organizations = client.get_organizations_in_my_collaboration()
org_ids = [organization["id"] for organization in organizations]


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
    # store responses
    local_parameters = results[:,0]
    acc_results[:, round] = results[:,1]
    dataset_sizes = results[:,2]

    parameters = average(local_parameters, dataset_sizes, None, dataset,use_sizes=False, use_imbalances=False)

    # do 'global' test
    testModel.set_params(parameters)
    global_acc_results[round]  = testModel.test(X_test, y_test, criterion)

if array_save:
    with open ("class_imb_no_comp_local.npy", 'wb') as f:
        np.save(f, acc_results)

    with open ("class_imb_no_comp_global.npy", 'wb') as f2:
        np.save(f2, global_acc_results)

print(acc_results)
print(global_acc_results)
#print(np.mean(acc_results, axis=0))
print("final runtime: ", (time.time() - start_time)/ 60)
x = np.arange(num_global_rounds)
plt.plot(x, np.mean(acc_results, axis=0))
plt.plot(x,global_acc_results)
plt.show()
