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


def get_model_hyperparams():
    experiments = exp.ExperimentsFolder()
    approach = experiments['Seq2seqDSG']
    results = []
    for iteration in approach.all_iterations():
        hyperparamters = iteration.hyperparameters
        if hyperparamters.path.exists():
            hyperparamters = hyperparamters.load()
            results.append(hyperparamters[-1])
    return results

def export_predictions(*names, export_path='export'):
    experiments = exp.ExperimentsFolder()
    approach = experiments['Seq2seqDSG']
    for iteration in approach.all_iterations():
        if iteration.experiment_path.name in names:
            hyperparameters = iteration.hyperparameters
            if hyperparameters.path.exists():
                hyperparams = hyperparameters.load()
                file = ez.File(
                    f"{export_path}/{iteration.experiment_path.name}"
                    f"/hyperparameters.pkl"
                )
                file.save(hyperparams)
                for prediction in iteration.all_predictions():
                    pred = prediction.load()
                    file = ez.File(
                        f"{export_path}/{iteration.experiment_path.name}/predictions"
                        f"/{prediction.path.stem}.pkl"
                    )
                    file.save(pred)

def model_summary():
    hyperparams = get_model_hyperparams()
    models = []
    for hyperparam in hyperparams:
        if 'data' in hyperparam:
            fields = [
                'experiment', 'iteration', 'data', 'checkpoint'
            ]
            model = {field: hyperparam[field] for field in fields}
            data = model['data']
            i, j = data.rfind('/')+1, data.rfind('.')
            model['data'] = data[i:j]
            model['experiment'] = textify(model['experiment'])
            model['checkpoint'] = textify(model['checkpoint'])
            models.append(model)
    df = pd.DataFrame.from_records(models).groupby(['experiment']).max()
    return df

def main():
    df = model_summary()
    print(df)
    export_predictions(
        'DaringWookiee',
        'ExoticTeth',
        'ExuberantAtollon',
        'FearlessBespin',
        'FearlessBodhi',
        'InfiniteAhsoka',
        'ThunderousScarif',
        'WiseCerea',
        'WisePorg',
        'UnforgettableChristophsis',
    )


if __name__ == '__main__':
    main()


