


def exec(model, dataset, si, ci, sca, lr='default', num_rounds = 100, num_runs = 4, seed_offset = 0, local_batch_amt = 1):
    args = {
        'model_choice' : model,
        'dataset' : dataset,
        'sample_imbalance' : si,
        'ci' : ci,
        'sca' : sca,
        'lr' : lr,
        'num_rounds' :num_rounds,
        'num_runs' : num_runs,
        'seed_offset' : seed_offset,
        'local_batch_amt' : local_batch_amt
        }
    arguments = []
    for arg in args.keys():
        arguments.append(arg)
        arguments.append(args[arg])


    execfile('nntest/researcher_cl.py')

'''list of tests to run (and their corresponding cl args):

CNN: 
    (MNIST_4class, ci, nc)
    A2
    fashion_MNIST

FNN:
    MNIST_4class, si, nc
    MNIST_4class, si, wc
    MNIST_4class, IID
    MNIST_4class, ci, nc
    MNIST_4class, ci, wc
    A2
    fashion_MNIST

GBDT:
    A2
    fashion_MNIST

LR:
    MNIST_4class, ci, scaf
    MNIST_4class, si nc
    MNIST_4class, si wc

SVM:
    MNIST_4class, si, nc
    MNIST_4class, si, wc
'''