import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import numpy.linalg as LA


#fedAvg implementation
def average(in_params, set_sizes, class_imbalances, dataset, model_choice, use_sizes= False, use_imbalances = False) :

    parameters = init_params(dataset, model_choice, True)

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
def scaffold(dataset,model_choice, global_parameters, local_parameters, c, old_local_c, local_c, lr, use_c = True):
    num_clients = local_parameters.size
    parameters = init_params(dataset, model_choice, True)
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
    elif dataset == 'MNIST_2class':
        if class_imbalance:
            datasets = ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_class_imbalance/MNIST_2Class_class_imbalance_client" + str(i) + ".csv" for i in range(10)]
        elif sample_imbalance:
            datasets =["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_Sample_Imbalance/MNIST_2Class_sample_imbalance_client" + str(i) + ".csv" for i in range(10)]
        else:
            datasets= ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_IID/MNIST_2Class_IID_client" + str(i) + ".csv" for i in range(10)]
    elif dataset == 'MNIST_4class':
        if class_imbalance:
            datasets = ["/home/swier/Documents/afstuderen/datasets/4Class_class_imbalance/MNIST_4Class_class_imbalance_client" + str(i) + ".csv" for i in range(10)]
        elif sample_imbalance:
            datasets = ["/home/swier/Documents/afstuderen/datasets/4Class_sample_imbalance/MNIST_4Class_sample_imbalance_client" + str(i) + ".csv" for i in range(10)]
        else:
            datasets = ["/home/swier/Documents/afstuderen/datasets/4Class_IID/MNIST_4Class_IID_client" + str(i) + ".csv" for i in range(10)]
    else :
        raise(ValueError("unknown dataset"))
    
    return datasets

def get_config(dataset, model_choice, num_clients, class_imbalance, sample_imbalance):
    datasets = get_datasets(dataset, class_imbalance, sample_imbalance)
    parameters = init_params(dataset, model_choice, False)
    c, ci = get_c(dataset, model_choice, num_clients)
    X_test, y_test = get_full_dataset(datasets, model_choice)
        

    return datasets, parameters, X_test, y_test, c, ci


def init_params(dataset, model_choice, zeros = True):
        ### set the parameters dictionary to all zeros before aggregating 
    if zeros:
        if dataset == 'banana' :
            parameters= {
            'lin_layers.0.weight' : torch.zeros((4,2), dtype=torch.double),
            'lin_layers.0.bias' : torch.zeros((4), dtype=torch.double),
            'lin_layers.1.weight' : torch.zeros((2,4), dtype=torch.double),
            'lin_layers.1.bias' : torch.zeros((2), dtype=torch.double)
        }
        elif dataset == 'MNIST':
            parameters= {
            'lin_layers.0.weight' : torch.zeros((100,28*28), dtype=torch.double),
            'lin_layers.0.bias' : torch.zeros((100), dtype=torch.double),
            'lin_layers.2.weight' : torch.zeros((10,100), dtype=torch.double),
            'lin_layers.2.bias' : torch.zeros((10), dtype=torch.double)
        }
        elif dataset == 'MNIST_2class':
            if model_choice == "FNN":
                parameters= {
                'lin_layers.0.weight' : torch.zeros((100,28*28), dtype=torch.double),
                'lin_layers.0.bias' : torch.zeros((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.zeros((2,100), dtype=torch.double),
                'lin_layers.2.bias' : torch.zeros((2), dtype=torch.double)
                }
            elif model_choice == "CNN":
                parameters = {
                    'conv_layers.0.weight': torch.zeros((1,1,3,3)),
                    'conv_layers.0.bias' : torch.zeros(1),
                    'lin_layers.0.weight' : torch.zeros((2, 196)),
                    'lin_layers.0.bias' : torch.zeros(2)
                }
        elif dataset == "MNIST_4class":
            if model_choice == "FNN":
                parameters= {
                'lin_layers.0.weight' : torch.zeros((100,28*28), dtype=torch.double),
                'lin_layers.0.bias' : torch.zeros((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.zeros((4,100), dtype=torch.double),
                'lin_layers.2.bias' : torch.zeros((4), dtype=torch.double)
                }
            elif model_choice == "CNN":
                parameters = {
                    'conv_layers.0.weight': torch.zeros((1,1,3,3)),
                    'conv_layers.0.bias' : torch.zeros(1),
                    'lin_layers.0.weight' : torch.zeros((4, 196)),
                    'lin_layers.0.bias' : torch.zeros(4)
                }
            else:
                raise ValueError("model selection not known")
    else:
        if dataset == 'banana':
            parameters= {
                'lin_layers.0.weight' : torch.randn((4,2), dtype=torch.double),
                'lin_layers.0.bias' : torch.randn((4), dtype=torch.double),
                'lin_layers.2.weight' : torch.randn((2,4), dtype=torch.double),
                'lin_layers.2.bias' : torch.randn((2), dtype=torch.double)
            }
        elif dataset == 'MNIST' : 
        # mnist parameters
            parameters= {
                'lin_layers.0.weight' : torch.randn((100,28*28), dtype=torch.double),
                'lin_layers.0.bias' : torch.randn((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.randn((10,100), dtype=torch.double),
                'lin_layers.2.bias' : torch.randn((10), dtype=torch.double)
            } 
        elif dataset == 'MNIST_2class':
            if model_choice == "FNN":
                parameters= {
                'lin_layers.0.weight' : torch.randn((100,28*28), dtype=torch.double),
                'lin_layers.0.bias' : torch.randn((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.randn((2,100), dtype=torch.double),
                'lin_layers.2.bias' : torch.randn((2), dtype=torch.double)
                }
            elif model_choice == "CNN":
                parameters = {
                    'conv_layers.0.weight': torch.randn((1,1,3,3)),
                    'conv_layers.0.bias' : torch.randn(1),
                    'lin_layers.0.weight' : torch.randn((2, 196)),
                    'lin_layers.0.bias' : torch.randn(2)
                }
        elif dataset == 'MNIST_4class':
            if model_choice == "FNN":
                parameters= {
                'lin_layers.0.weight' : torch.randn((100,28*28), dtype=torch.double),
                'lin_layers.0.bias' : torch.randn((100), dtype=torch.double),
                'lin_layers.2.weight' : torch.randn((4,100), dtype=torch.double),
                'lin_layers.2.bias' : torch.randn((4), dtype=torch.double)
                }
            elif model_choice == "CNN":
                parameters = {
                    'conv_layers.0.weight': torch.randn((1,1,3,3)),
                    'conv_layers.0.bias' : torch.randn(1),
                    'lin_layers.0.weight' : torch.randn((4, 196)),
                    'lin_layers.0.bias' : torch.randn(4)
                }
    return (parameters)

def get_full_dataset(datasets, model_choice):
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
    if model_choice == "CNN":
        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    return X_test, y_test

def get_c(dataset, model_choice, num_clients):
    c = init_params(dataset, model_choice, zeros=True)

    ci = [c] * num_clients
    return c, ci

class heatmap():
    def __init__(self, num_clients, num_rounds):
        self.map = np.empty((num_clients, num_rounds))
        self.num_clients = num_clients
        self.num_rounds = num_rounds


    def save_round(self, round, client_params, global_param_dict, is_dict = True):
        if is_dict:
            param_size = self.calc_param_size(global_param_dict)
            global_arr = self.dict_to_arr(param_size, global_param_dict)
        for client_idx, client in enumerate(client_params):
            if is_dict:
                client_arr = self.dict_to_arr(param_size, client)
            else: 
                client_arr = client
                global_arr = global_param_dict
            self.map[client_idx, round] =  LA.norm(global_arr - client_arr)

    def save_round_arr(self, round, client_params, global_params):
        for client_idx, client in enumerate(client_params):
            self.map[client_idx, round] = LA.norm(global_params - client)

    def dict_to_arr(self, arr_size, dict):
        pointer = 0
        return_array = np.zeros((arr_size))
        for key in dict.keys():
            tmp_arr = dict[key].detach().numpy().reshape(-1)
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


    def show_map(self, title = "", show_text=False):
        fig, ax = plt.subplots()
        final_map = self.map / LA.norm(self.map, axis=0)
        #print(self.map)
        #print(LA.norm(self.map, axis=0))
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

        if show_text:
            for i in range(self.map.shape[0]):
                for j in range(self.map.shape[1]):
                    text = ax.text(j,i, round(final_map[i,j], 2), ha="center", va="center", color="b")
            
        #print(LA.norm(self.map))
        plt.xlabel("rounds")
        plt.ylabel("clients")
        plt.title(title)
        plt.colorbar(im, shrink=0.5)
        plt.show()

    def save_map(self, path):
        with open(path, 'wb') as f:
            np.save(f, self.map)


def get_save_str(m_choice, c_i, s_i, u_sc, u_si, lr,  epoch, batch):
    if c_i:
        str1 = "ci"
    elif s_i:
        str1 = "si"
    else:
        str1 = "IID"

    if u_sc:
        str2 = "scaf"
    elif u_si:
        str2 = "size_comp"
    else:
        str2 = "no_comp"

    
    return (str1 + "_" + str2 + "_" + m_choice + "_lr" + str(lr) + "_lepo" + str(epoch) + "_ba" + str(batch))
    
