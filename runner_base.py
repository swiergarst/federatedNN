import torch
import numpy as np


import sqlite3

from v6_simpleNN_py.model import model

from vantage6.client import Client



class runner_base(object):

    def __init__(self):
        # client-side specific vars (that probably need to be sent to client)
        self.set_client_vars()

        # dataset-specific vars
        self.set_dataset_vars()

        # fed algorithm-specifi vars
        self.set_algo_vars()

        # experiment-specific vars
        self.set_experiment_vars()

    def set_client_vars(self, lr = 5e-1, lepochs=1, lbatch=1):
        self.client_vars = {
            "lr" : lr,
            "lepochs" : lepochs,
            "lbatch" : lbatch
        }

    def set_dataset_vars(self, dataset = "2class_MNIST", class_imbalance = False, sample_imbalance = False):
        self.dataset = dataset
        self.class_imbalance = class_imbalance
        self.sample_imbalance = sample_imbalance

    def set_algo_vars(self, model_choice = "FNN", use_scaffold = False, use_fADMM = False, early_stopping = False, stopping_threshold = 10, lr_global = 1, rho = 0.4):
        self.model_choice = model_choice
        self.use_scaffold = use_scaffold
        self.use_fADMM = use_fADMM
        self.early_stopping = early_stopping
        self.stopping_threshold = stopping_threshold
        self.lr_global = lr_global
        self.rho = rho
    
    def set_experiment_vars(self,image, num_global_rounds = 100, num_clients = 10, num_runs = 4, seed_offset = 0):
        self.image = image
        self.num_global_rounds = num_global_rounds
        self.num_clients = num_clients
        self.num_runs = num_runs
        self.seed_offset = seed_offset

    def run(self):
        self.connect()

        ids = [org['id'] for org in self.client.collaboration.get(1)['organizations']]
        prefix = self.get_save_str()

        for run in range(self.num_runs):
            seed = self.set_seed(run)
            model = self.create_model()

            # init parameters and data structures
            if self.use_scaffold:
                c = self.init_global_params(zeros=True)
                ci = [self.init_global_params(zeros=True)] * self.num_clients
                ci = np.array(ci)
                old_ci = np.array([c.copy()] * self.num_clients)
    
            global_params = self.init_params()
            local_params = np.array([global_params] * self.num_clients)
            acc_results = np.zeros((self.num_clients, self.num_global_rounds))

            for round in range(self.num_global_rounds):
                for i in range(self.num_clients):
                    old_ci[i] = ci[i].copy()
                print("run", run, "round", round)
                # send out task and retrieve results
                if self.use_scaffold:
                    task_list = self.send_task_scaffold(ids)
                    results = self.get_task_results_scaffold(task_list)
                else:
                    task_id = self.send_task(ids)
                    results = self.get_task_results(task_id)
                # put results in the right structures
                acc_results[round,:] = results[0,:]
                local_params = results[1,:]

                # do averaging
                if self.use_scaffold:
                    global_params, c = self.scaffold(global_params, local_params, ci, old_ci, c, self.lr_global   )
                elif self.use_fADMM:
                    y = results[2,:]
                    global_params = self.fADMM(global_params, local_params, y)
                else:
                    set_sizes = results[2,:]
                    global_params = self.average(local_params, set_sizes)
                
                #clean up database once in a while
                if (round%10 == 0):
                    self.clear_database()
            
            # save files
            with open (prefix + "seed" + str(seed) + ".npy") as f:
                np.save(f, acc_results)


    

    def average(self, in_params, set_sizes):
        #create size-based weights
        weights = np.ones_like(set_sizes)/self.num_clients
        total_size = np.sum(set_sizes) 
        weights = set_sizes / total_size
    
        #do averaging
        if isinstance(in_params[0], np.ndarray):
            parameters = np.zeros_like(in_params[0])
            for i in range (in_params.shape[1]):
                for j in range(self.num_clients):
                    parameters[i] += weights[j] * in_params[j,i]
        else:
            parameters = self.init_params(self.dataset, self.model_choice, True)

            for param in parameters.keys():
                for i in range(self.num_clients):
                    parameters[param] += weights[i] * in_params[i][param]

        return parameters

    def scaffold(self, global_parameters, local_parameters, c, old_local_c, local_c, lr, use_c = True, key = None):
        
        #for sklearn-based implementations
        if isinstance(global_parameters, np.ndarray):
            num_clients = local_parameters.shape[0]

            param_agg = np.zeros_like(global_parameters)
            parameters = np.zeros_like(global_parameters)
            c_agg = np.zeros_like(global_parameters)
            
            for i in range(num_clients):
                c_agg += local_c[i][key] - old_local_c[i][key]
                param_agg += local_parameters[i] - global_parameters
            parameters = global_parameters + (lr/num_clients) * param_agg
            c[key] = c[key] + (1/num_clients) * c_agg

        #for pytorch-based implementations
        else:
            num_clients = local_parameters.size
            parameters = self.init_global_params(zeros = True)
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

    def fADMM(self, W, y):
        num_clients = len(y)
        W_g = self.params_to_numpy(self.init_global_params(True))

        for para in W_g.keys():

            par_list = np.array([W[i][para].numpy() for i in range(num_clients)])
            y_list = np.array([(1/self.rho) * y[i][para] for i in range(num_clients)])
            W_g[para] = np.sum((par_list + y_list), axis = 0)/num_clients    
        return W_g


    def get_save_str(self):
        if self.class_imbalance:
            str1 = "ci"
        elif self.sample_imbalance:
            str1 = "si"
        else:
            str1 = "IID"

        if self.use_scaffold:
            str2 = "scaf"
        else:
            str2 = "fedAvg"

        return (self.dataset + str1 + "_" + str2 + "_" + self.model_choice + "_lr" + str(self.lr_local) + "_lepo" + str(self.lepochs) + "_ba" + str(self.lbatch))
   
    def connect(self):
        print("Attempt login to Vantage6 API")
        self.client = Client("http://localhost", 5000, "/api")
        self.client.authenticate("researcher", "1234")
        privkey = "/home/swier/.local/share/vantage6/node/privkey_testOrg0.pem"
        self.client.setup_encryption(privkey)

    def clear_database(self):
        con = sqlite3.connect("/home/swier/Documents/afstuderen/default.sqlite")

        cur = con.cursor()

        com1 = "PRAGMA foreign_keys = 0;"
        com2 = "DELETE FROM task;"
        com3 = "DELETE FROM result;"# LIMIT 100;"

        cur.execute(com1)
        cur.execute(com2)
        cur.execute(com3)


        con.commit()
        con.close()

    def params_to_numpy(self, parameters):
        raise(NotImplementedError("implement params_to_numpy"))

    def set_seed(self, run):
        raise(NotImplementedError("implement set_seed"))

    def create_model(self):
        raise(NotImplementedError("implement create model"))

    def init_global_params(self):
        raise(NotImplementedError("implement init_global_params"))

    def send_task(self, ids):
        raise(NotImplementedError("implement send_task"))

    def send_task_scaffold(self, ids):
        raise(NotImplementedError("implement send_task_scaffold"))

    def get_task_results(self):
        raise(NotImplementedError("implement get_task_results"))

    def get_task_results_scaffold(self, task_list):
        raise(NotImplementedError("implement get_task_results_scaffold"))