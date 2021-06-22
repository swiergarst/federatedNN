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
from helper_functions import average, get_datasets, get_config
start_time = time.time()
### connect to server


print("Attempt login to Vantage6 API")
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "1234")

client.setup_encryption(None)


#organizations = client.get_organizations_in_my_collaboration()
#org_ids = [organization["id"] for organization in organizations]




### parameter settings

#torch

criterion = nn.CrossEntropyLoss()
optimizer = 'SGD'


ids = [org['id'] for org in client.collaboration.get(1)['organizations']]

#dataset
dataset = 'MNIST_2class_IID'
class_imbalance = False
sample_imbalance = False


#federated settings
num_global_rounds = 100
num_clients = 10
num_runs = 4

# arrays to store results
acc_results = np.zeros((num_runs, num_clients, num_global_rounds))
complete_test_results = np.empty((num_runs, num_global_rounds))




### main loop
for run in range(num_runs):
    torch.manual_seed(run)
    datasets, parameters, X_test, y_test = get_config(dataset,class_imbalance, sample_imbalance)
    #test model for global testing
    testModel = model(dataset)
    testModel.double()
    for round in range(num_global_rounds):
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
                    'dataset' : dataset
                }
            },
            name = "class imbalance, no comp round " + str(round),
            image = "sgarst/federated-learning:2ClassNN1",
            organization_ids=ids,
            collaboration_id= 1
        )
        #print(round_task)
        info("Waiting for results")
        res = client.get_results(task_id=round_task.get("id"))
        attempts=1
        #print(res)
        while(None in [res[i]["result"] for i in range(num_clients)]  and attempts < 20):
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
        acc_results[run, :, round] = np.array(results[:,1])
        dataset_sizes = np.array(results[:,2])
        parameters = average(local_parameters, dataset_sizes, None, dataset, use_imbalances=False, use_sizes=False)

        # 'global' test
        testModel.set_params(parameters)
        complete_test_results[run ,round]  = testModel.test(X_test, y_test, criterion)


### save arrays to files
with open ("IID_no_comp_local_seed03.npy", 'wb') as f:
    np.save(f, acc_results)

with open ("IID_no_comp_global_seed03.npy", 'wb') as f2:
    np.save(f2, complete_test_results)

print(repr(acc_results))
print(repr(complete_test_results))
#print(np.mean(acc_results, axis=0))
print("final runtime", (time.time() - start_time)/60)
x = np.arange(num_global_rounds)
#plt.plot(x, np.mean(acc_results, axis=0))
#plt.plot(x, complete_test_results)
#plt.show()
    ### generate new model parameters
