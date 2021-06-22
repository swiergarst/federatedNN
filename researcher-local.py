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
from helper_functions import average, get_datasets, get_config, scaffold

start_time = time.time()

# parameters

#torch 
torch.manual_seed(1)
criterion = nn.CrossEntropyLoss()
optimizer = 'SGD'
lr_local = 5e-1
lr_global = 5e-1
use_scaffold = True
use_c = True
#c = np.zeros(4)

#federated settings
num_global_rounds = 5
num_clients = 10


#dataset
dataset = 'MNIST_2class_IID'
class_imbalance = False
sample_imbalance = False

datasets, parameters, X_test, y_test, c, ci = get_config(dataset, num_clients, class_imbalance, sample_imbalance)

#test model for global testing
testModel = model(dataset)
testModel.double()

# arrays to store results
acc_results = np.zeros((num_clients, num_global_rounds))
global_acc_results = np.zeros((num_global_rounds))
array_save = False
param_log_local = np.zeros((num_clients, num_global_rounds, 2) )
param_log_global = np.zeros((num_global_rounds, 2))

### connect to server
client = ClientMockProtocol(
    datasets= datasets,
    module="v6_simpleNN_py"
### connect to server
)
organizations = client.get_organizations_in_my_collaboration()
org_ids = [organization["id"] for organization in organizations]


### main loop
for round in range(num_global_rounds):
    old_ci = ci
    ### request task from clients
    round_task = client.create_new_task(
        input_= {
            'method' : 'train_and_test',
            'kwargs' : {
                'parameters' : parameters,
                'criterion': criterion,
                'optimizer': optimizer,
                'lr' : lr_local,
                'scaffold' : use_scaffold,
                'c' : c, 
                'dataset' : dataset, 
                'use_c' : use_c
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
    ci = results[:,3]
    for i in range(num_clients):
        param_log_local[i, round, :] = local_parameters[i]['fc2.bias'].numpy()


    if use_scaffold:
        parameters, c = scaffold(dataset, parameters, local_parameters, c, old_ci, ci, lr_global, use_c = use_c)
        #parameters = average(local_parameters, dataset_sizes, None, dataset,use_sizes=False, use_imbalances=False)
    else:
        parameters = average(local_parameters, dataset_sizes, None, dataset,use_sizes=False, use_imbalances=False)

    param_log_global[round, :] = parameters['fc2.bias'].numpy()
    # do 'global' test
    testModel.set_params(parameters)
    global_acc_results[round]  = testModel.test(X_test, y_test, criterion)

if array_save:
    with open ("class_imb_no_comp_local.npy", 'wb') as f:
        np.save(f, acc_results)

    with open ("class_imb_no_comp_global.npy", 'wb') as f2:
        np.save(f2, global_acc_results)

#print(acc_results)
#print(global_acc_results)
#print(ci)
#print(np.mean(acc_results, axis=0))
#print(repr(param_log_global))
#print(repr(param_log_local))
print("final runtime: ", (time.time() - start_time)/ 60)
x = np.arange(num_global_rounds)
plt.plot(x, np.mean(acc_results, axis=0))
plt.plot(x,global_acc_results)
plt.show()
'''
fig1 = plt.figure(1)
fig1.plot(x, param_log_global[:,0])
fig1.plot(x, param_log_global[:,1])
fig1.title("global")
fig1.show()


fig2 = plt.figure(2)
fig2.plot(x, np.mean(param_log_local, axis = 0)[:,0])
fig2.plot(x, np.mean(param_log_local, axis = 0)[:,1])
fig2.title("local")
fig2.show()
'''