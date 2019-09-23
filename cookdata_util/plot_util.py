__all__ = ('plot_value_on_bar', 'plot_grid_search')

import matplotlib.pyplot as plt
import seaborn as sns
import json
from .constants import CONSTANTS, check_constants

def plot_value_on_bar(y, ax):
    for index, value in enumerate(y):
        ax.text(index, value, value, ha='center', va='bottom')

def plot_grid_search(gs):
    first_param = gs.cv_results_['params'][0]
    if len(first_param > 1):
        raise ValueError("Only 1d param grid is supported")
    param_name = [*first_param.keys()][0]

    fig, ax = plt.subplots(figsize=(8, 6))

    X = gs.cv_results_['param_' + param_name]
    Y = gs.cv_results_['mean_test_score']
    sns.lineplot(x=X, y=Y, ax=ax)

    for x, y in zip(X, Y):
        ax.text(x, y, str(round(y, 3)).replace('0.', '.'), va='bottom', ha='center')

    ax.set_title(f'不同 {param_name} 的分数图')
    ax.set_xlabel(param_name)
    ax.set_ylabel('分数')
