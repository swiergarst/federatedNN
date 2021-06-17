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
from helper_functions import average, get_datasets

start_time = time.time()
# parameters
criterion = nn.CrossEntropyLoss()
optimizer = 'SGD'

num_global_rounds = 100
num_clients = 10
dataset = 'MNIST_2class_IID'
class_imbalance = True
sample_imbalance = False
   
datasets = get_datasets(dataset, class_imbalance, sample_imbalance)   
### connect to server
client = ClientMockProtocol(
    #datasets= ["/home/swier/Documents/afstuderen/nnTest/v6-simpleNN-py/local/banana/banana_dataset_client" + str(i) + ".csv" for i in range(10)],
    
    datasets= datasets,
    module="v6_simpleNN_py"
)

organizations = client.get_organizations_in_my_collaboration()
org_ids = [organization["id"] for organization in organizations]





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
elif dataset == 'MNIST_2class_IID': 
    parameters= {
        'fc1.weight' : torch.randn((100,28*28), dtype=torch.double),
        'fc1.bias' : torch.randn((100), dtype=torch.double),
        'fc2.weight' : torch.randn((2,100), dtype=torch.double),
        'fc2.bias' : torch.randn((2), dtype=torch.double)
    }

# get the dataset for 'global' testing
if dataset == 'MNIST' : 
    MNIST_test = torch.load("/home/swier/Documents/afstuderen/MNIST/processed/test.pt")
    X_test = MNIST_test[0].flatten(start_dim=1)/255
    y_test = MNIST_test[1]
elif dataset == 'MNIST_2class_IID' :
    #datasets = get_datasets(dataset, class_imbalance, sample_imbalance)
    #datasets = ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_IID/MNIST_2Class_IID_client" + str(i) + ".csv" for i in range(10)]   
    dim_num = 784
    dims = ['pixel' + str(i) for i in range(dim_num)]

    for i,  set in enumerate(datasets):
        data = pd.read_csv(set)
        X_test_partial = data.loc[data['test/train'] == 'test'][dims].values
        y_test_partial = data.loc[data['test/train'] == 'test']['label'].values
        if i == 0:
            X_test = X_test_partial
            y_test = y_test_partial
        else:
            X_test_arr = np.concatenate((X_test, X_test_partial))
            y_test_arr = np.concatenate((y_test, y_test_partial))

    X_test = torch.as_tensor(X_test_arr, dtype=torch.double)
    y_test = torch.as_tensor(y_test_arr, dtype=torch.int64)

testModel = model(dataset)
testModel.double()
acc_results = np.zeros((num_clients, num_global_rounds))
global_acc_results = np.zeros((num_global_rounds))
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
    dataset_sizes = results[:,2]
    ### set the parameters dictionary to all zeros before aggregating
    parameters = average(local_parameters, dataset_sizes, None, dataset, use_imbalances=False, use_sizes=True)

    # do 'global' test

    testModel.set_params(parameters)
    global_acc_results[round]  = testModel.test(X_test, y_test, criterion)

with open ("class_imb_no_comp_local.npy", 'wb') as f:
    np.save(f, acc_results)

with open ("class_imb_no_comp_global.npy", 'wb') as f2:
    np.save(f2, global_acc_results)

#print(acc_results)
#print(complete_test_results)
#print(np.mean(acc_results, axis=0))
print("final runtime: ", (time.time() - start_time)/ 60)
x = np.arange(num_global_rounds)
plt.plot(x, np.mean(acc_results, axis=0))
plt.plot(x,global_acc_results)
plt.show()
