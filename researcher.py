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

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from v6_simpleNN_py.model import model
from io import BytesIO
from vantage6.tools.util import info
from vantage6.client import Client
from helper_functions import average
start_time = time.time()
### connect to server


print("Attempt login to Vantage6 API")
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "1234")

client.setup_encryption(None)


#organizations = client.get_organizations_in_my_collaboration()
#org_ids = [organization["id"] for organization in organizations]




### parameter settings

#architecture array will have the following format:
# [layer_0_input_dim, layer_1_input_dim, .. , layer_n_input_dim, layer_n_output_dim]
# so if the array is of length 2 there will be a single layer, 3 params equals 2 layers with the second param the size of the second layer, etc
# this is not counting the output as a layer    

architecture = np.array([2,4,2])
criterion = nn.CrossEntropyLoss()
optimizer = 'SGD'


ids = [i for i in range(1,11)]

num_global_rounds = 100
num_clients = 10
dataset = 'MNIST'

torch.manual_seed(42)
#create the weights and biases
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
elif dataset == 'MNIST_2class_IID' : 
    parameters = {
        'fc1.weight' : torch.randn((100,28*28), dtype=torch.double),
        'fc1.bias' : torch.randn((100), dtype=torch.double),
        'fc2.weight' : torch.randn((2,100), dtype=torch.double),
        'fc2.bias' : torch.randn((2), dtype=torch.double)
    }


acc_results = np.zeros((num_clients, num_global_rounds))
complete_test_results = np.empty((1, num_global_rounds))
### create a model for 'global' testing
### also get the full testing data
# TODO: rewrite this to a function
if dataset == 'MNIST' : 
    MNIST_test = torch.load("/home/swier/Documents/afstuderen/MNIST/processed/test.pt")
    X_test = MNIST_test[0].flatten(start_dim=1)/255
    y_test = MNIST_test[1]
elif dataset == 'MNIST_2class_IID' :
    datasets = ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_IID/MNIST_2Class_IID_client" + str(i) + ".csv" for i in range(10)]   
    dim_num = 784
    dims = ['pixel' + str(i) for i in range(dim_num)]
    for set in datasets:
        X_test_partial = set.loc[set['test/train'] == 'test']

testModel = model(dataset)


### main loop
for round in range(num_global_rounds):
    print("starting round", round)
    ### request task from clients
    round_task = client.post_task(
        input_= {
            'method' : 'train_and_test',
            'kwargs' : {
                'architecture' : architecture,
                'parameters' : parameters,
                'criterion': criterion,
                'optimizer': optimizer,
                'dataset' : dataset
            }
        },
        name = "nntest round " + str(round),
        image = "sgarst/federated-learning:nnTest",
        organization_ids=ids,
        collaboration_id= 1
    )
    info("Waiting for results")
    res = client.get_results(task_id=round_task.get("id"))
    attempts=1
    
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
    acc_results[:, round] = np.array(results[:,1])
    dataset_sizes = np.array(results[:,2])
    parameters = average(local_parameters, dataset_sizes, None, dataset, use_imbalances=False, use_sizes=False)

    # 'global' test
    testModel.set_params(parameters)
    complete_test_results[0,round]  = testModel.test(X_test, y_test, criterion)


### save arrays to files
with open ("testfile.npy", 'wb') as f:
    np.save(f, acc_results)

with open ("testfile2.npy", 'wb') as f2:
    np.save(f2, complete_test_results)

print(repr(acc_results))
print(repr(complete_test_results))
#print(np.mean(acc_results, axis=0))
print("final runtime", time.time() - start_time)
x = np.arange(num_global_rounds)
plt.plot(x, np.mean(acc_results, axis=0))
plt.plot(x, complete_test_results)
plt.show()
    ### generate new model parameters
