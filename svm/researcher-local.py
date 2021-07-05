from vantage6.tools.mock_client import ClientMockProtocol
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from helper_functions import get_datasets, heatmap


dataset = "MNIST_2class_IID"
### connect to server
datasets = get_datasets(dataset)
datasets.remove("/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_IID/MNIST_2Class_IID_client9.csv")
client = ClientMockProtocol(
    datasets= datasets,
    module="v6_svm_py"
### connect to server
)
organizations = client.get_organizations_in_my_collaboration()
org_ids = [organization["id"] for organization in organizations]

save_file  = True


lr = 0.5
num_runs = 1
num_global_rounds = 100
avg_coef = np.zeros((1,784))
avg_intercept = np.zeros((1))
parameters = [avg_coef, avg_intercept]
num_clients = 9
accuracies = np.zeros((num_runs, num_clients, num_global_rounds))
coefs = np.zeros((num_clients, 784))
intercepts = np.zeros((num_clients, 1))



for run in range(num_runs):
    seed = run
    map = heatmap(num_clients, num_global_rounds )
    for round in range(num_global_rounds):
        round_task = client.create_new_task(
            input_= {
                'method' : 'train_and_test',
                'kwargs' : {
                    'parameters' : parameters,
                    'seed' : seed
                    }
            },
            organization_ids=org_ids
        )
        ## aggregate responses
        results = np.array(client.get_results(round_task.get("id")))
        accuracies[run, :,round] = results[:, 0]
        coefs = results[:,1]
        intercepts = results[:,2]

        #avg_coef, _ = scaffold(avg_coef, coefs, None, None, None, 0.5, use_c=False)
        intercept_agg = 0
        coef_agg = np.zeros((784))
        for i in range(num_clients):
            intercept_agg += intercepts[i] - avg_intercept
            coef_agg +=  intercepts[i] - avg_intercept

        avg_intercept = avg_intercept + (lr / num_clients) * intercept_agg
        avg_coef = avg_coef + (lr / num_clients) * coef_agg

        #avg_coef = np.mean(coefs, axis=0)
        #avg_intercept = np.mean(intercepts, axis=0)
        map.save_round(round, coefs, avg_coef, is_dict=False)
        parameters = [avg_coef, avg_intercept]
    map.save_map("../w10/simulated_svm_avg_no9_seed" + str(seed) + "map.npy")

if save_file:
    ### save arrays to files
    with open ("../w10/simulated_svm_iid_avg_no9_seed" + str(seed)+ ".npy", 'wb') as f:
        np.save(f, accuracies)




#print(repr(accuracies))
#plt.plot(np.arange(num_global_rounds), accuracies.T)
#plt.show()
#map.show_map("SVM classifier, IID datasets")
#map.save_map("../w10/simulated_svm_average_seed" + str(seed) + "map.npy")