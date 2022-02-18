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
from config_functions import get_datasets, get_config,get_full_dataset
from comp_functions import average, scaffold

start_time = time.time()

# parameters

#torch 
torch.manual_seed(1)
criterion = nn.CrossEntropyLoss()
optimizer = 'SGD'
lr_local = 5e-1
lr_global = 5e-1
local_epochs = 1
local_batch_amt = 1

# boolean settings
save_file = False
class_imbalance = False
sample_imbalance = False
use_scaffold = False
use_c = True
use_sizes = False
#c = np.zeros(4)

#federated settings
num_global_rounds = 10
num_clients = 10
num_runs = 1
seed_offset = 0


#dataset
dataset = 'MNIST_2class' #options: MNIST_2class, MNIST, MNIST_4class, fashion_MNIST, A2
model_choice = 'FNN'
datasets, parameters, X_test, y_test, c, ci = get_config(dataset, model_choice, num_clients, class_imbalance, sample_imbalance)
ci = np.array(ci)
#test model for global testing
testModel = model(dataset, model_choice, c)
testModel.double()

# arrays to store results
acc_results = np.zeros((num_clients, num_global_rounds))
global_acc_results = np.zeros((num_global_rounds))
param_log_local = np.zeros((num_clients, num_global_rounds, 2) )
param_log_global = np.zeros((num_global_rounds, 2))

### connect to server
client = ClientMockProtocol(
    datasets= datasets,
    module="v6_simpleNN_py"
)
'''
A_alt = np.array([[0,1,9],
                [1,0,2],
                [2,1,3],
                [3,2,4],
                [4,3,5],
                [5,4,6],
                [6,5,7],
                [7,6,8],
                [8,7,9],
                [9,8,0]])
'''
A_alt = np.array([[0],
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
                [7],
                [8],
                [9]])


organizations = client.get_organizations_in_my_collaboration()
org_ids = [organization["id"] for organization in organizations]

local_parameters = np.array([parameters]*num_clients)

### main loop
for round in range(num_global_rounds):
    print("round: ", round)
    ### request task from clients
    task_list = np.empty(num_clients, dtype=object)
    
    for i, org_id in enumerate(org_ids):
        parameters = local_parameters[A_alt[i,:]]

        round_task = client.create_new_task(
            input_= {
                'method' : 'train_and_test',
                'kwargs' : {
                    'parameters' : local_parameters[i],
                    'nb_parameters' : parameters,
                    'criterion': criterion,
                    'optimizer': optimizer,
                    'model_choice' : model_choice,
                    'lr' : lr_local,
                    'local_epochs' : local_epochs,
                    'local_batch_amt' : local_batch_amt,
                    'dataset' : dataset, 
                    }
            },
            organization_ids=[org_ids[i]]
        )
        task_list[i] =  round_task

    finished = False
    local_parameters = np.empty(num_clients, dtype=object)
    dataset_sizes = np.empty(num_clients, dtype = object)
    while (finished == False):
        for task_i, task in enumerate(task_list):
            result = np.array(client.get_results(task.get("id")))
            #print(result[0,0])
            local_parameters[task_i] = result[0,0]
            acc_results[task_i, round] = result[0,1]
            dataset_sizes[task_i] = result[0,2]
            ci[task_i] = result[0,3]
        finished = True
    
    #print(org_ids)
    

    '''
    for i in range(num_clients):
        param_log_local[i, round, :] = local_parameters[i]['fc2.bias'].numpy()
    '''
    '''
    round_task = client.create_new_task(
            input_= {
                'method' : 'train_and_test',
                'kwargs' : {
                    'parameters' : parameters,
                    'criterion': criterion,
                    'optimizer': optimizer,
                    'model_choice' : model_choice,
                    'lr' : lr_local,
                    'local_epochs' : local_epochs,
                    'local_batch_amt' : local_batch_amt,
                    'scaffold' : use_scaffold,
                    'c' : c, 
                    'dataset' : dataset, 
                    'use_c' : use_c,
                    'ci' : ci[0]
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
    '''

if save_file:
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
#plt.plot(x, acc_results.T)
plt.plot(x, np.mean(acc_results, axis=0))
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