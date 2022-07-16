import numpy as np
import pickle


def save_data(dataloader, name):
    X, Y = [], []
    for data, target in dataloader:
        X.append(data.detach().cpu().numpy())
        Y.append(target.detach().cpu().numpy())
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    data = {}
    data['X'] = X[:600]
    data['Y'] = Y[:600]

    with open("processeData/{}.pkl".format(name), "wb") as pkl_file:
        pickle.dump(data, pkl_file)
