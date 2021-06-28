import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import numpy.linalg as LA


#fedAvg implementation
def average(in_params, set_sizes, class_imbalances, dataset, use_sizes= False, use_imbalances = False) :

    parameters = init_params(dataset, True)

    #create size-based weights
    num_clients = set_sizes.size
    weights = np.ones_like(set_sizes)/num_clients

    
    if use_sizes:
        total_size = np.sum(set_sizes) 
        weights = set_sizes / total_size
    
    #do averaging
    for param in parameters.keys():
        for i in range(num_clients):
            parameters[param] += weights[i] * in_params[i][param]

    return parameters

#scaffold implementation
def scaffold(dataset, global_parameters, local_parameters, c, old_local_c, local_c, lr, use_c = True):
    num_clients = local_parameters.size
    parameters = init_params(dataset,  True)
    for param in parameters.keys():
        param_agg = torch.clone(parameters[param])
        c_agg = torch.clone(param_agg)
        #calculate the sum of differences between the local and global model
        for i in range(num_clients):
            param_agg += local_parameters[i][param] - global_parameters[param]
            c_agg += local_c[i][param] - old_local_c[i][param]
        #calculate new weight value
        parameters[param] = global_parameters[param] + (lr/num_clients) * param_agg 
        if use_c:
            c[param] = c[param] + (1/num_clients) * c_agg
    return parameters, c

def get_datasets(dataset, class_imbalance = False, sample_imbalance = False):
    if dataset == 'banana':
        datasets =  ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/banana/banana_dataset_client" + str(i) + ".csv" for i in range(10)]
    elif dataset == 'MNIST':
        datasets= ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST/MNIST_dataset_client" + str(i) + ".csv" for i in range(10)]
    elif dataset == 'MNIST_2class_IID':
        if class_imbalance:
            datasets = ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_class_imbalance/MNIST_2Class_class_imbalance_client" + str(i) + ".csv" for i in range(10)]
        elif sample_imbalance:
            datasets =["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_Sample_Imbalance/MNIST_2Class_sample_imbalance_client" + str(i) + ".csv" for i in range(10)]
        else:
            datasets= ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_IID/MNIST_2Class_IID_client" + str(i) + ".csv" for i in range(10)]
    return datasets

def get_config(dataset, num_clients, class_imbalance, sample_imbalance):
    datasets = get_datasets(dataset, class_imbalance, sample_imbalance)
    parameters = init_params(dataset, False)
    c, ci = get_c(parameters, num_clients)

    if dataset == 'MNIST_2class_IID':
        X_test, y_test = get_full_dataset(datasets)
        

    return datasets, parameters, X_test, y_test, c, ci


def init_params(dataset, zeros = True):
        ### set the parameters dictionary to all zeros before aggregating 
    if zeros:
        if dataset == 'banana' :
            parameters= {
            'fc1.weight' : torch.zeros((4,2), dtype=torch.double),
            'fc1.bias' : torch.zeros((4), dtype=torch.double),
            'fc2.weight' : torch.zeros((2,4), dtype=torch.double),
            'fc2.bias' : torch.zeros((2), dtype=torch.double)
        }
        elif dataset == 'MNIST':
            parameters= {
            'fc1.weight' : torch.zeros((100,28*28), dtype=torch.double),
            'fc1.bias' : torch.zeros((100), dtype=torch.double),
            'fc2.weight' : torch.zeros((10,100), dtype=torch.double),
            'fc2.bias' : torch.zeros((10), dtype=torch.double)
        }
        elif dataset == 'MNIST_2class_IID':
            parameters= {
            'fc1.weight' : torch.zeros((100,28*28), dtype=torch.double),
            'fc1.bias' : torch.zeros((100), dtype=torch.double),
            'fc2.weight' : torch.zeros((2,100), dtype=torch.double),
            'fc2.bias' : torch.zeros((2), dtype=torch.double)
        }
    else:
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
    return (parameters)

def get_full_dataset(datasets):
    dim_num = 784
    dims = ['pixel' + str(i) for i in range(dim_num)]

    for i,  set in enumerate(datasets):
        data = pd.read_csv(set)
        X_test_partial = data.loc[data['test/train'] == 'test'][dims].values
        y_test_partial = data.loc[data['test/train'] == 'test']['label'].values
        if i == 0:
            X_test_arr = X_test_partial
            y_test_arr = y_test_partial
        else:
            X_test_arr = np.concatenate((X_test_arr, X_test_partial))
            y_test_arr = np.concatenate((y_test_arr, y_test_partial))

    X_test = torch.as_tensor(X_test_arr, dtype=torch.double)
    y_test = torch.as_tensor(y_test_arr, dtype=torch.int64)

    return X_test, y_test

def get_c(parameters, num_clients):
    c = {
        'fc1.weight' : torch.zeros_like(parameters['fc1.weight']),
        'fc1.bias' : torch.zeros_like(parameters['fc1.bias']),
        'fc2.weight' : torch.zeros_like(parameters['fc2.weight']),
        'fc2.bias' : torch.zeros_like(parameters['fc2.bias'])
    }
    ci = [c] * num_clients
    return c, ci

class heatmap():
    def __init__(self, num_clients, num_rounds):
        self.map = np.empty((num_clients, num_rounds))
        self.num_clients = num_clients
        self.num_rounds = num_rounds


    def save_round(self, round, client_params, global_param_dict):
        param_size = self.calc_param_size(global_param_dict)
        global_arr = self.dict_to_arr(param_size, global_param_dict)
        for client_idx, client in enumerate(client_params):
            client_arr = self.dict_to_arr(param_size, client)
            self.map[client_idx, round] =  LA.norm(global_arr - client_arr)

    def dict_to_arr(self, arr_size, dict):
        pointer = 0
        return_array = np.zeros((arr_size))
        for key in dict.keys():
            tmp_arr = dict[key].numpy().reshape(-1)
            return_array[pointer:pointer+tmp_arr.size] = tmp_arr
            pointer += tmp_arr.size
        return return_array
    
    def calc_param_size(self, param_dict):
        size = 0
        for key in param_dict.keys():
            key_size = 1
            for dict_size in param_dict[key].size():
                key_size *= dict_size
            size += key_size
        return(size)


    def show_map(self):
        fig, ax = plt.subplots()
        final_map = self.map / LA.norm(self.map, axis=0)
        print(self.map)
        print(LA.norm(self.map, axis=0))
        im = ax.imshow(final_map)
        ax.set_xticks(np.arange(self.map.shape[1]))
        ax.set_yticks(np.arange(self.map.shape[0]))
        xlabels = np.arange(self.num_rounds)
        ylabels = ["client" + str(i) for i in range(self.num_clients)]
        ax.set_xticklabels(xlabels)
        ax.set_yticklabels(ylabels)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                text = ax.text(j,i, round(final_map[i,j], 2), ha="center", va="center", color="b")
        
        #print(LA.norm(self.map))
        plt.show()

        def save_map(self, path):
            with open(path) as f:
                np.save(f, self.map)