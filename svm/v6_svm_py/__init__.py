from sklearn.linear_model import SGDClassifier
import pandas as pd












def master():
    pass

def RPC_train_and_test(data, parameters):
    model = SGDClassifier(loss="hinge", penalty="l2", max_iter = 1, warm_start=True, fit_intercept=True)

    model.coef_ = parameters[0]
    model.intercept_ = parameters[1]


    dim_num = 784
    dims = ['pixel' + str(i) for i in range(dim_num)]
    X_train_arr = data.loc[data['test/train'] == 'train'][dims].values
    y_train_arr = data.loc[data['test/train'] == 'train']['label'].values
    X_test_arr = data.loc[data['test/train'] == 'test'][dims].values
    y_test_arr = data.loc[data['test/train'] == 'test']['label'].values

    model.fit(X_train_arr, y_train_arr)

    result = model.score(X_test_arr, y_test_arr)

    return(result, model.coef_, model.intercept_)