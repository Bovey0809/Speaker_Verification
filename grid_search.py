from sklearn.model_selection import ParameterGrid
from train import train


def pipeline(**params):
    # training
    train('preprocessed_VCTK',
          lr=params['lr'],
          epochs=params['max_epochs'],
          proj=params['proj'],
          hidden=params['hidden'],
          num_layers=params['num_layers'],
          opt=params['opt'],
          N=params['N'],
          step_size=params['step_size'])
    pass


if __name__ == '__main__':

    # parameters sets
    # OPTIMAL EPOCHS
    # VCTK: 4e3

    parameters = {
        'max_epochs': [4000],
        'lr': [0.0001],
        'proj': [512],
        'hidden': [32],
        'num_layers': [3],
        'opt': ['Adam'],
        'N': [64],
        'step_size': [1e7]
    }
    grid = ParameterGrid(parameters)

    for i, params in enumerate(grid):
        pipeline(lr=params['lr'],
                 max_epochs=params['max_epochs'],
                 proj=params['proj'],
                 hidden=params['hidden'],
                 num_layers=params['num_layers'],
                 opt=params['opt'],
                 N=params['N'],
                 step_size=params['step_size'])
