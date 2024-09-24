
import ezpyz as ez
from ezpyz.expydite import explore as ex
import pandas as pd
import random
import dst.approaches.dst_seq_data as dstseq
import dst.analysis.experiments_folder as exp
import pickle
from pprint import pprint

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

textify = lambda s: ''.join([c for c in s if c.isalnum() or c.isspace() or c in ['-', '_']])


def get_results():
    experiments = exp.ExperimentsFolder()
    approaches = experiments.values()
    results = []
    for approach in approaches:
        for iteration in approach.all_iterations():
            for result in iteration.all_results():
                result = result.dict()
                results.append(result)
    return results

def to_df(result_dicts, *columns):
    ascending = None
    descending = None
    cols = []
    for columns_ in columns:
        for column in columns_.split(','):
            column = column.strip()
            if column:
                if column.startswith('^'):
                    column = column[1:]
                    ascending = column
                elif column.endswith('^'):
                    column = column[:-1]
                    ascending = column
                elif column.startswith('*'):
                    column = column[1:]
                    descending = column
                elif column.endswith('*'):
                    column = column[:-1]
                    descending = column
                cols.append(column)
    if cols:
        result_dicts = [{
            col: result_dict.get(col) for col in cols
        } for result_dict in result_dicts]
    df = pd.DataFrame(result_dicts)
    if ascending:
        df = df.sort_values(ascending, ascending=True)
    if descending:
        df = df.sort_values(descending, ascending=False)
    return df


def get_best(df, groupby='experiment', metric='slot_update_accuracy', better='higher'):
    if better not in ['higher', 'lower']:
        raise ValueError("The 'better' parameter must be 'higher' or 'lower'.")
    ascending = better == 'lower'
    sorted_df = df.groupby(groupby, group_keys=False).apply(
        lambda x: x.sort_values(metric, ascending=ascending)
    )
    best_rows = sorted_df.groupby(groupby).first().sort_values(metric, ascending=ascending)
    return best_rows


def leaderboard(results, data_name, approach='all', *metrics, better='higher'):
    if not metrics:
        metrics = ['slot_update_accuracy', 'joint_goal_accuracy']
    results = [
        result for result in results
        if result['data_name'] == data_name and (
            approach == 'all'
            or textify(result['approach']) == approach
        )
    ]
    results = to_df(results, f'approach, experiment, epoch, *{", ".join(metrics)}')
    results = get_best(results, 'experiment', metrics[0], better)
    return results

def details(results, approach, experiment, epoch, data_name):
    newline = '\n'
    newline_repr = repr(newline)
    for result in results:
        if all([
            textify(result['approach']) == approach,
            textify(result['experiment']) == experiment,
            str(result['epoch']) == epoch,
            result['data_name'] == data_name,
        ]):
            display = '\n'.join([
                f'{key}: {str(value).replace(newline, newline_repr):.50s}'
                for key, value in result.items()
            ])
            return display


results = get_results()
lb = leaderboard(
    results, 'gptdst5k_valid_domains_0', 'all', 'slot_update_accuracy', 'joint_goal_accuracy'
)
print(lb, '\n\n')

print(details(results, 'Seq2seqDst', 'IntrepidWicket', '4', 'gptdst5k_valid_domains_0'))

