__all__ = ('init_pd_plt',)
import pandas as pd
import matplotlib.pyplot as plt


def init_pd_plt():
    # 显示数据时显示所有列
    pd.options.display.max_columns = None

    # 设置字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['font.size'] = 10
