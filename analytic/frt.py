from .draw_utils import type_check, get_tp_fp
from matplotlib import pyplot as plt
import matplotlib as mp
import numpy as np


def fraud_ratio_table(y_hat, y, ax, split_n=10, table=False):
    y_hat = type_check(y_hat)
    y = type_check(y)

    if len(y_hat) != len(y):
        raise Exception('y_hat and y is supposed to have same length')
        return None

    split_n = len(y) // 20 if split_n > len(y) / 20 else split_n

    tmp = get_tp_fp(y, y_hat, split_n)
    bad_ratio = tmp[1] / (tmp[0] + tmp[1])
    section_count = tmp[0] + tmp[1]
    mid_score = list(tmp.reset_index().bin.map(lambda x: int(x.mid)))
    c_map = 1 - bad_ratio

    # Colorize the graph based on likeability:
    likeability_scores = np.array(c_map)
    data_normalizer = mp.colors.Normalize()
    color_map = mp.colors.LinearSegmentedColormap(
        "my_map",
        {
            "red": [(0, 0, 0), (1, 0.65, 0)],
            "green": [(0, 0, 0.2), (1, 0.88, 0)],
            "blue": [(0, 0, 0), (1, 0, 0)]
        }
    )

    # Map xs to numbers:
    x_nums = np.arange(1, len(section_count)+1)
    # Plot a bar graph:
    ax.bar(
        x_nums,
        section_count,
        align="center",
        color=color_map(data_normalizer(likeability_scores)),
        alpha=0.8
    )

    plt.xticks(x_nums, mid_score)

    ax2 = ax.twinx()
    ax2.plot(
        x_nums,
        bad_ratio,
        linestyle='-',
        color='dodgerblue',
        marker='o',
        markeredgecolor='red',
        markeredgewidth=1,
        markerfacecolor='snow')
    if table:
        tmp = get_tp_fp(y, y_hat, split_n)
        tmp['section_count_ratio'] = (tmp[1] + tmp[0]) / (sum(tmp[0]) + sum(tmp[1]))
        tmp['good_count'] = tmp[0]
        tmp['bad_count'] = tmp[1]
        tmp['total_count'] = tmp[0] + tmp[1]
        tmp['section_good_ratio'] = tmp[0] / (tmp[1] + tmp[0])
        tmp['section_bad_ratio'] = tmp[1] / (tmp[1] + tmp[0])
        tmp['good_cum'] = tmp[0].cumsum()
        tmp['bad_cum'] = tmp[1].cumsum()
        tmp['section_ratio_cum'] = tmp['section_count_ratio'].cumsum()
        tmp['good_cum_ratio'] = tmp['good_cum'] / tmp['good_cum'][-1]
        tmp['bad_cum_ratio'] = tmp['bad_cum'] / tmp['bad_cum'][-1]
        tmp['ks'] = tmp['bad_cum_ratio'] - tmp['good_cum_ratio']

        res = tmp.reset_index()[
            ['bin',
             'good_count',
             'bad_count',
             'total_count',
             'section_count_ratio',
             'section_good_ratio',
             'section_bad_ratio',
             'section_ratio_cum',
             'good_cum_ratio',
             'bad_cum_ratio',
             'ks']
        ]
        res.columns = [
            'Score Range',
            'Good Count',
            'Bad Count',
            'Total Count',
            'Proportion',
            'Good Rate',
            'Bad Rate',
            'Cumulative of Total Proportion',
            'Cumulative of Good Proportion',
            'Cumulative of Bad Proportion',
            'ks'
        ]
        return res
