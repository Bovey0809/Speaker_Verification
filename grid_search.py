import sys
import pandas as pd
from sklearn.model_selection import ParameterGrid
from test import test
from train import train
from time import ctime


def pipeline(**params):
    # training
    model_path = train(lr=params['lr'],
                       epochs=params['max_epochs'],
                       proj=params['proj'],
                       hidden=params['hidden'],
                       num_layers=params['num_layers'],
                       opt=params['opt'],
                       N=params['N'],
                       step_size=params['step_size'])

    # testing
    # acc = test(model_path,
    #            proj=params['proj'],
    #            hidden=params['hidden'],
    #            num_layers=params['num_layers'])
    pass


if __name__ == '__main__':

    # parameters sets
    parameters = {
        'max_epochs': [1500],
        'lr': [0.005, 0.01],
        'proj': [128],
        'hidden': [768],
        'num_layers': [3],
        'opt': ['SGD'],
        'N': [128],
        'step_size': [1e2, 1e3, 1e4]
    }
    grid = ParameterGrid(parameters)
    max_acc = 0
    result = pd.DataFrame(grid)

    for i, params in enumerate(grid):
        pipeline(lr=params['lr'],
                 max_epochs=params['max_epochs'],
                 proj=params['proj'],
                 hidden=params['hidden'],
                 num_layers=params['num_layers'],
                 opt=params['opt'],
                 N=params['N'],
                 step_size=params['step_size'])
    #     print(f"ACC: {acc:4f}")
    #     result.loc[i, 'acc'] = acc
    #     if acc > max_acc:
    #         max_acc = acc
    #         print(f"MAX ACC: {max_acc:4f}")
    #         best_param = params
    # print(f"max_acc: {max_acc}")
    # result.to_csv(f'{ctime()}log.csv')
