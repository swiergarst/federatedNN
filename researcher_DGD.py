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




### parameter settings ###

#torch
criterion = nn.CrossEntropyLoss()
optimizer = 'SGD'
lr_local = 5e-2
lr_global = 1 #only affects scaffold. 1 is recommended

local_epochs = 1 #local epochs between each communication round
local_batch_amt = 10 #amount of  batches the data gets split up in at each client   

ids = [org['id'] for org in client.collaboration.get(1)['organizations']]

#dataset and booleans
dataset = 'MNIST_2class' # options: MNIST_2class, MNIST_4class, MNIST, fashion_MNIST, A2, 3node, 2node
week = "datafiles/w27/"

model_choice = "CNN" #decides the neural network; either FNN or CNN
save_file = True # whether to save results in .npy files

# these settings change the distribution of the datasets between clients. sample_imbalance is not checked if class_imbalance is set to true
class_imbalance = False
sample_imbalance = True

use_scaffold= False # if true, uses scaffold instead of federated averaging
use_c = True # if false, all control variates are kept 0 in SCAFFOLD (debug purposes)
use_sizes = True # if false, the non-weighted average is used in federated averaging (instead of the weighted average)

#federated settings
num_global_rounds = 100 #number of communication rounds
num_clients = 10 #number of clients (make sure this matches the amount of running vantage6 clients)
num_runs = 3 #amount of experiments to run using consecutive seeds
seed_offset = 1 #decides which seeds to use: seed = seed_offset + current_run_number

### end of settings ###

prefix = get_save_str(dataset, model_choice, class_imbalance, sample_imbalance, use_scaffold, use_sizes, lr_local, local_epochs, local_batch_amt)




prevmap = heatmap(num_clients, num_global_rounds)
newmap = heatmap(num_clients, num_global_rounds)
#if use_scaffold:
cmap = heatmap(num_clients , num_global_rounds)
c_log = np.zeros((num_global_rounds))
ci_log = np.zeros((num_global_rounds, num_clients))

#quick hack b/c i was too lazy to update the model part of the image
if dataset == "2node":
    dataset_tosend = "3node"
else :
    dataset_tosend = dataset


# DGD stuff
#connectivity matrix

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

for run in range(num_runs):
    acc_results = np.zeros((num_clients, num_global_rounds))

    seed = run + seed_offset
    torch.manual_seed(seed)
    datasets, parameters, X_test, y_test, c, ci = get_config(dataset, model_choice, num_clients, class_imbalance, sample_imbalance)

    parameters_full = np.array([parameters]*num_clients)

    for round in range(num_global_rounds):
        task_list = np.empty(num_clients, dtype=object)

        for i, org_id in enumerate(ids[0:num_clients]):
            parameters = parameters_full[A_alt[i,:]]

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
                    'dataset' : dataset_tosend,    
                    }
            },
            name =  prefix + ", round " + str(round),
            image = "sgarst/federated-learning:fedNN10",
            organization_ids=[org_id],
            collaboration_id= 1
                
            )
        task_list[i] =  round_task

        finished = False
        parameters_full = np.empty(num_clients, dtype=object)
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
                        parameters_full[task_i] = res[0]
                        acc_results[task_i, round] = res[1]
                        solved_tasks.append(task_i)
            
            #task_list = np.copy(new_task_list)
            if not (None in parameters_full):
                finished = True
            print("waiting")
            time.sleep(1)


    if save_file:
        #prevmap.save_map(week + prefix + "prevmap_seed" + str(seed) + ".npy")
        #newmap.save_map(week + prefix + "newmap_seed" + str(seed) + ".npy")
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
for i in range (num_clients):#
   plt.plot(x, acc_results[i,:])
legend = ["client " + str(i) for i in range(num_clients)]
legend.append("full")
plt.plot(x, complete_test_results)
plt.legend(legend)
plt.show()

plt.plot(x, c_log)
plt.plot(x, ci_log)
#plt.show()
    ### generate new model parameters
