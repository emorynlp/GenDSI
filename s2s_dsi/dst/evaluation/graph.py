import matplotlib.pyplot as plt
import numpy as np


def create_bar_chart_with_confidence_intervals_mod(title, xs, xtitle=None, ytitle=None, /, **kwargs):
    num_series = len(kwargs)
    series_length = len(next(iter(kwargs.values())))

    # Create an array with the positions of each bar on the X axis
    x_pos = np.arange(series_length)

    # Width of a bar
    bar_width = 0.8 / num_series

    for i, (name, data) in enumerate(kwargs.items()):
        # Split the tuples into separate lists
        points, ci_lower, ci_upper = zip(*data)

        ci_upper = [ci_upper[i] - points[i] for i in range(len(points))]
        ci_lower = [points[i] - ci_lower[i] for i in range(len(points))]

        # Create the bars with the 'points' height and named label
        # Adjust the position of this series
        plt.bar(
            x_pos + i * bar_width, points, yerr=[ci_lower, ci_upper],
            align='center', alpha=0.8, ecolor='black', capsize=10,
            label=name, width=bar_width
        )

    # Add labels and title
    plt.ylabel(ytitle)
    plt.xlabel(xtitle)
    plt.title(title)

    # Create names on the x-axis and adjust their position
    plt.xticks(x_pos + bar_width * (num_series - 1) / 2, [str(x) for x in xs])

    if title == 'Correctness':
        plt.ylim(0.6, 1.0)

    # Add legend
    plt.legend()

    plt.savefig(f'figs/{title}.png', dpi=1024)

    # Show graphic
    plt.show()


def graph_human_eval_results():
    # Data for Completeness on GenDIS
    completeness_gen_dis = {
        "SGD-DSG": (0.323, 0.273, 0.378),
        "GPTPipe": (0.933, 0.899, 0.957),
        "E2E-DSG": (0.957, 0.927, 0.975)
    }

    # Data for Completeness on SGD-X dialogues
    completeness_sgd_x = {
        "T5-SGDXF": (0.693, 0.639, 0.743),
        "GPTPipe": (0.900, 0.860, 0.929),
        "E2E-DSG": (0.947, 0.915, 0.968)
    }

    # Data for Correctness on GenDIS
    correctness_gen_dis = {
        "SGD-DSG": (0.726, 0.670, 0.776),
        "GPTPipe": (0.819, 0.794, 0.842),
        "E2E-DSG": (0.811, 0.786, 0.835)
    }

    # Data for Correctness on SGD-X dialogues
    correctness_sgd_x = {
        "SGD-DSG": (0.908, 0.874, 0.933),
        "GPTPipe": (0.847, 0.794, 0.870),
        "E2E-DSG": (0.817, 0.789, 0.842)
    }

    xs = ['SGD-DSG', 'GPTPipe', 'E2E-DSG']
    gendis_correctness = list(correctness_gen_dis.values())
    sgd_correctness = list(correctness_sgd_x.values())
    gendis_completeness = list(completeness_gen_dis.values())
    sgd_completeness = list(completeness_sgd_x.values())

    create_bar_chart_with_confidence_intervals_mod(
        'Correctness', xs, 'Model', 'Correctness',
        DSG5K=gendis_correctness, SGD=sgd_correctness
    )

    create_bar_chart_with_confidence_intervals_mod(
        'Completeness', xs, 'Model', 'Completeness',
        DSG5K=gendis_completeness, SGD=sgd_completeness
    )


if __name__ == '__main__':
    graph_human_eval_results()