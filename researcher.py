# with lots of inspiration from the courselabs from the CS4240 deep learning course and fedML

### imports
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import os
import time 
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from v6_simpleNN_py.model import model
from io import BytesIO
from vantage6.tools.util import info
from vantage6.client import Client
from helper_functions import average, get_datasets, get_config, scaffold, heatmap, get_save_str
start_time = time.time()
### connect to server


print("Attempt login to Vantage6 API")
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "1234")
privkey = "/home/swier/.local/share/vantage6/node/privkey_testOrg0.pem"
client.setup_encryption(privkey)


#organizations = client.get_organizations_in_my_collaboration()
#org_ids = [organization["id"] for organization in organizations]




### parameter settings

#torch

criterion = nn.CrossEntropyLoss()
optimizer = 'SGD'
lr_local = 5e-1
lr_global = 5e-1
local_epochs = 1
local_batch_amt = 1

ids = [org['id'] for org in client.collaboration.get(1)['organizations']]

#dataset and booleans
dataset = 'MNIST_2class'
week = "../datafiles/w13/"

model_choice = "FNN"
save_file = True
class_imbalance = True
sample_imbalance = False
use_scaffold=True
use_c = True
use_sizes = True
prefix = get_save_str(dataset, model_choice, class_imbalance, sample_imbalance, use_scaffold, use_sizes, lr_local, local_epochs, local_batch_amt)

#federated settings
num_global_rounds = 100
num_clients = 10
num_runs = 3
seed_offset = 1


prevmap = heatmap(num_clients, num_global_rounds)
newmap = heatmap(num_clients, num_global_rounds)
#if use_scaffold:
cmap = heatmap(num_clients , num_global_rounds)
### main loop
for run in range(num_runs):
    # arrays to store results
    acc_results = np.zeros((num_clients, num_global_rounds))
    complete_test_results = np.empty((num_global_rounds))

    seed = run + seed_offset
    torch.manual_seed(seed)
    datasets, parameters, X_test, y_test, c, ci = get_config(dataset, model_choice, num_clients, class_imbalance, sample_imbalance)
    #test model for global testing
    testModel = model(dataset, model_choice)
    testModel.double()
    for round in range(num_global_rounds):
        old_ci = ci.copy() 
        #old_ci = ci
        #print("initial old ci: ", old_ci)
        print("starting round", round)
        ### request task from clients
        round_task = client.post_task(
            input_= {
                'method' : 'train_and_test',
                'kwargs' : {
                    #'architecture' : architecture,
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
                    'use_c' : use_c
                }
            },
            name =  prefix + ", round " + str(round),
            image = "sgarst/federated-learning:2ClassNN5",
            organization_ids=ids,
            collaboration_id= 1
        )

        #print(round_task)
        info("Waiting for results")
        res = client.get_results(task_id=round_task.get("id"))
        attempts=1
        #print(res)
        while(None in [res[i]["result"] for i in range(num_clients)]  and attempts < 20000):
            print("waiting...")
            time.sleep(1)
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
        dataset_sizes = np.array(results[:,2])
        prevmap.save_round(round, local_parameters, parameters)

        if use_scaffold:
            ci = results[:,3]

            parameters, c = scaffold(dataset, model_choice, parameters, local_parameters, c, old_ci, ci, lr_global, use_c = use_c)
            #print("old ci: ", old_ci)
            cmap.save_round(round, ci, c)
        else:
            parameters = average(local_parameters, dataset_sizes, None, dataset, model_choice, use_imbalances=False, use_sizes= use_sizes)


        newmap.save_round(round, local_parameters, parameters)
        # 'global' test
        testModel.set_params(parameters)
        complete_test_results[round]  = testModel.test(X_test, y_test, criterion)
    
    
    if save_file:
        if use_scaffold:    
            cmap.save_map(week + prefix + "cmap_seed" + str(seed) + ".npy")
        prevmap.save_map(week + prefix + "prevmap_seed" + str(seed) + ".npy")
        newmap.save_map(week + prefix + "newmap_seed" + str(seed) + ".npy")
        ### save arrays to files
        with open (week + prefix + "local_seed" + str(seed) + ".npy", 'wb') as f:
            np.save(f, acc_results)

        with open (week + prefix + "_global_seed"+ str(seed) + ".npy", 'wb') as f2:
            np.save(f2, complete_test_results)


print(repr(acc_results))
print(repr(complete_test_results))
#print(np.mean(acc_results, axis=0))
print("final runtime", (time.time() - start_time)/60)
x = np.arange(num_global_rounds)
cmap.show_map(normalized=False)
#prevmap.show_map()
#newmap.show_map()

#plt.plot(x, np.mean(acc_results, axis=1, keepdims=False)[0,:])
plt.plot(x, complete_test_results)
plt.show()

    ### generate new model parameters
