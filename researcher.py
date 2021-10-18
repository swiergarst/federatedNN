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
from helper_functions import heatmap
from config_functions import get_config, clear_database, get_save_str
from comp_functions import average, scaffold
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
lr_local = 5e-3
lr_global = 1
local_epochs = 1

local_batch_amt = 1

ids = [org['id'] for org in client.collaboration.get(1)['organizations']]

#dataset and booleans
dataset = '3node' # options: MNIST_2class, MNIST_4class, MNIST, fashion_MNIST, A2
week = "datafiles/w23/"

model_choice = "FNN"
save_file = False
class_imbalance = False
sample_imbalance = False
use_scaffold= False
use_c = True
use_sizes = True
prefix = get_save_str(dataset, model_choice, class_imbalance, sample_imbalance, use_scaffold, use_sizes, lr_local, local_epochs, local_batch_amt)

#federated settings
num_global_rounds = 20
num_clients = 10
num_runs = 1
seed_offset = 0


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
    ci = np.array(ci)
    old_ci = np.array([c.copy()] * num_clients)

    #test model for global testing
    testModel = model(dataset, model_choice, c)
    testModel.double()
    for round in range(num_global_rounds):
        for i in range(num_clients):
            old_ci[i] = ci[i].copy()
        #old_ci = ci
        #print("initial old ci: ", old_ci)
        print("starting round", round)

        task_list = np.empty(num_clients, dtype=object)
        

        for i, org_id in enumerate(ids):
            #print("org id \t ids[i]")
            #print(org_id, "\t", ids[i])

            round_task = client.post_task(
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
                        'ci': ci[i],
                        'dataset' : dataset, 
                        'use_c' : use_c
                        }
                },
                name =  prefix + ", round " + str(round),
                image = "sgarst/federated-learning:fedNN9",
                organization_ids=[org_id],
                collaboration_id= 1
                
            )
            task_list[i] =  round_task

        finished = False
        local_parameters = np.empty(num_clients, dtype=object)
        dataset_sizes = np.empty(num_clients, dtype = object)
        while (finished == False):
            #new_task_list = np.copy(task_list)
            solved_tasks = []
            for task_i, task in enumerate(task_list):
                result = client.get_results(task_id = task.get("id"))
                #print(result)
                if not (None in [result[0]["result"]]):
                #print(result[0,0])
                    if not (task_i in solved_tasks):
                        res = (np.load(BytesIO(result[0]["result"]),allow_pickle=True))
                        #print(res)
                        local_parameters[task_i] = res[0]
                        acc_results[task_i, round] = res[1]
                        dataset_sizes[task_i] = res[2]
                        ci[task_i] = res[3]
                        solved_tasks.append(task_i)
            
            #task_list = np.copy(new_task_list)
            if not (None in local_parameters):
                finished = True
            print("waiting")
            time.sleep(1)

        if use_scaffold:
            #ci = results[:,3]

            parameters, c = scaffold(dataset, model_choice, parameters, local_parameters, c, old_ci, ci, lr_global, use_c = use_c)
            #print("old ci: ", old_ci)
            cmap.save_round(round, ci, c)
        else:
            parameters = average(local_parameters, dataset_sizes, None, dataset, model_choice, use_imbalances=False, use_sizes= use_sizes)


        newmap.save_round(round, local_parameters, parameters)
        # 'global' test
        testModel.set_params(parameters)
        complete_test_results[round]  = testModel.test(X_test, y_test, criterion)
        #if (round % 10) == 0:
        #    clear_database()
    
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
            # clear database every 10 rounds



print(repr(acc_results))
print(repr(complete_test_results))
#print(np.mean(acc_results, axis=0))
print("final runtime", (time.time() - start_time)/60)
x = np.arange(num_global_rounds)
#cmap.show_map(normalized=False)
#prevmap.show_map()
#newmap.show_map()

#plt.plot(x, np.mean(acc_results, axis=1, keepdims=False)[0,:])
plt.plot(x, complete_test_results)
plt.show()

    ### generate new model parameters
