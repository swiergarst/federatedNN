import numpy as np
import pandas as pd
import torch



def average(in_params, set_sizes, class_imbalances, dataset, use_sizes= False, use_imbalances = False) :

    ### set the parameters dictionary to all zeros before aggregating 
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

def get_config(dataset, class_imbalance, sample_imbalance):
    if dataset == 'banana':
        datasets =  ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/banana/banana_dataset_client" + str(i) + ".csv" for i in range(10)]
        parameters= {
            'fc1.weight' : torch.randn((4,2), dtype=torch.double),
            'fc1.bias' : torch.randn((4), dtype=torch.double),
            'fc2.weight' : torch.randn((2,4), dtype=torch.double),
            'fc2.bias' : torch.randn((2), dtype=torch.double)
        }
        #TODO: create X_test and y_test for banana dataset
    elif dataset == 'MNIST' : 
    # mnist parameters
        datasets= ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST/MNIST_dataset_client" + str(i) + ".csv" for i in range(10)]
        parameters= {
            'fc1.weight' : torch.randn((100,28*28), dtype=torch.double),
            'fc1.bias' : torch.randn((100), dtype=torch.double),
            'fc2.weight' : torch.randn((10,100), dtype=torch.double),
            'fc2.bias' : torch.randn((10), dtype=torch.double)
        }    
        MNIST_test = torch.load("/home/swier/Documents/afstuderen/MNIST/processed/test.pt")
        X_test = MNIST_test[0].flatten(start_dim=1)/255
        y_test = MNIST_test[1]
    elif dataset == 'MNIST_2class_IID':
        parameters= {
            'fc1.weight' : torch.randn((100,28*28), dtype=torch.double),
            'fc1.bias' : torch.randn((100), dtype=torch.double),
            'fc2.weight' : torch.randn((2,100), dtype=torch.double),
            'fc2.bias' : torch.randn((2), dtype=torch.double)
        }
        if class_imbalance:
            datasets = ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_class_imbalance/MNIST_2Class_class_imbalance_client" + str(i) + ".csv" for i in range(10)]
        elif sample_imbalance:
            datasets =["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_Sample_Imbalance/MNIST_2Class_sample_imbalance_client" + str(i) + ".csv" for i in range(10)]
        else:
            datasets= ["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_IID/MNIST_2Class_IID_client" + str(i) + ".csv" for i in range(10)]
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

    return datasets, parameters, X_test, y_test