from vantage6.tools.mock_client import ClientMockProtocol
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from helper_functions import get_datasets


dataset = "MNIST_2class_IID"
### connect to server
client = ClientMockProtocol(
    datasets= get_datasets(dataset),
    module="v6_svm_py"
### connect to server
)
organizations = client.get_organizations_in_my_collaboration()
org_ids = [organization["id"] for organization in organizations]

num_global_rounds = 100
parameters = [np.zeros((1,784)), np.zeros((1))]
num_clients = 10
accuracies = np.zeros((num_clients, num_global_rounds))
coefs = np.zeros((num_clients, 784))
intercepts = np.zeros((num_clients, 1))
for round in range(num_global_rounds):
    round_task = client.create_new_task(
        input_= {
            'method' : 'train_and_test',
            'kwargs' : {
                'parameters' : parameters,
                }
        },
        organization_ids=org_ids
    )
    ## aggregate responses
    results = np.array(client.get_results(round_task.get("id")))
    accuracies[:,round] = results[:, 0]
    coefs = results[:,1]
    intercepts = results[:,2]

    avg_coef = np.mean(coefs, axis=0)
    avg_intercept = np.mean(intercepts, axis=0)
    parameters = [avg_coef, avg_intercept]

print(repr(accuracies))
plt.plot(np.arange(num_global_rounds), accuracies.T)
plt.show()