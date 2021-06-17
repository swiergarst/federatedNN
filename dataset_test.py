import pandas as pd

from helper_functions import get_datasets

datasets = get_datasets('MNIST_2class_IID', False, True)
#datasets =["/home/swier/Documents/afstuderen/nnTest/v6_simpleNN_py/local/MNIST_2Class_Sample_Imbalance/MNIST_2Class_sample_imbalance_client" + str(i) + ".csv" for i in range(10)]
 
for set in datasets:
    set_df = pd.read_csv(set)
    print(set_df.shape)