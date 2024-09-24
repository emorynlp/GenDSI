
import ezpyz as ez
import pathlib as pl
import dst.data.dst_data as dst

def to_train_split(name):
    return name.replace('valid', 'train').replace('test', 'train')
def to_valid_split(name):
    if 'train_train' in name:
        return name.replace('train_train', 'valid_train')
    return name.replace('train', 'valid').replace('test', 'valid')
def to_test_split(name):
    if 'train_train' in name:
        return name.replace('train_train', 'test_train')
    return name.replace('train', 'test').replace('valid', 'test')
def to_split(name, split):
    if split == 'train':
        return to_train_split(name)
    elif split == 'valid':
        return to_valid_split(name)
    elif split == 'test':
        return to_test_split(name)


class IterationResultFolder:
    def __init__(self, iteration, data_name):
        if not isinstance(iteration, IterationFolder):
            iteration = IterationFolder(iteration)
        self.iteration: IterationFolder = iteration
        self.data_name = data_name

    @property
    def path(self):
        return self.iteration.path

    @property
    def experiment_path(self):
        return self.iteration.experiment_path

    @property
    def experiment_folder(self):
        return self.iteration.experiment_folder

    @property
    def model_path(self):
        return self.iteration.model_path

    @property
    def prediction_path(self):
        return self.iteration.prediction_path

    @property
    def hyperparameters(self):
        return self.iteration.hyperparameters

    def results(self, split='valid'):
        return ez.Cache(self.path / f'{to_train_split(self.data_name)}.{split}_results.pkl')

    def predictions(self, split='valid'):
        data_name = to_split(self.data_name, split)
        return ez.Cache(self.iteration.prediction_path / f'{data_name}.dstseqdata')

    def dict(self, split='valid'):
        entries = {}
        if self.hyperparameters.path.exists():
            hyperparameters = self.hyperparameters.load()
            if hyperparameters:
                entries.update(hyperparameters[-1])
        elif self.iteration.experiment_folder.hyperparameters.path.exists():
            hyperparameters = self.iteration.experiment_folder.hyperparameters.load()
            if hyperparameters:
                entries.update(hyperparameters[-1])
        results = self.results(split)
        if results is not None and results.path.exists():
            results = results.load()
            entries.update(vars(results))
        if 'epoch' not in entries or entries.get('epoch') is None:
            entries['epoch'] = self.iteration.path.stem
        entries['data_name'] = self.data_name
        entries.setdefault('epoch', self.iteration.path.stem)
        return entries

class IterationFolder(dict[str,IterationResultFolder]):
    def __init__(self, path):
        self.path:pl.Path = ez.Cache(path).path
        self.experiment_path:pl.Path = self.path.parent
        self.model_path:pl.Path = ez.Cache(path).path / 'model'
        self.prediction_path:pl.Path = ez.Cache(path).path / 'predictions'
        self.hyperparameters:ez.Cache = ez.Cache(self.path/'hyperparameters.pkl')
        data_names = set()
        if self.prediction_path.exists():
            for path in self.prediction_path.iterdir():
                if path.is_file():
                    data_names.add(path.stem)
        if self.path.exists():
            for path in self.path.iterdir():
                if path.is_file():
                    data_names.add(path.name.split('.', 1)[0])
        self.update({
            data_name: IterationResultFolder(self, data_name)
            for data_name in data_names if data_name != 'hyperparameters'
        })
        self.experiment_folder = None

    def all_results(self, data_name=None):
        if data_name is None:
            return [
                result for result in self.values()
                if result.results().path.exists()
            ]
        return [
            result for result in self.values()
            if result.data_name == data_name and result.results().path.exists()
        ]

    def all_predictions(self):
        predictions = []
        if self.prediction_path.exists():
            for prediction in self.prediction_path.iterdir():
                if prediction.is_file():
                    predictions.append(prediction)
        return [ez.Cache(prediction) for prediction in predictions]


class ExperimentFolder(dict[str,IterationFolder]):
    def __init__(self, path):
        self.path:pl.Path = ez.Cache(path).path
        self.hyperparameters:ez.Cache = ez.Cache(self.path/'0'/'hyperparameters.pkl')
        self.update({
            iteration_folder.name: IterationFolder(iteration_folder)
            for iteration_folder in self.path.iterdir()
            if iteration_folder.is_dir()
        })
        for iteration in self.values():
            iteration.experiment_folder = self

    def all_iterations(self):
        return list(self.values())

    def all_results(self, data_name=None):
        return [
            iteration[data_name]
            for iteration in self.values()
            if data_name in iteration
        ]

    def all_predictions(self):
        return [
            prediction for iteration in self.all_iterations()
            for prediction in iteration.all_predictions()
        ]

class ApproachFolder(dict[str,ExperimentFolder]):
    def __init__(self, path):
        self.path:pl.Path = ez.Cache(path).path
        self.update({
            experiment_folder.name: ExperimentFolder(experiment_folder)
            for experiment_folder in self.path.iterdir()
            if experiment_folder.is_dir()
        })

    def all_experiments(self):
        return list(self.values())

    def all_iterations(self):
        return [
            result for experiment in self.all_experiments()
            for result in experiment.all_iterations()
        ]

    def all_results(self, data_name=None):
        return [
            result for experiment in self.all_experiments()
            for result in experiment.all_results(data_name)
        ]

    def all_predictions(self):
        return [
            prediction for experiment in self.all_experiments()
            for prediction in experiment.all_predictions()
        ]

class ExperimentsFolder(dict[str,ApproachFolder]):
    def __init__(self, folder='ex'):
        self.path:pl.Path = ez.Cache(folder).path
        self.update({
            approach_folder.name: ApproachFolder(approach_folder)
            for approach_folder in self.path.iterdir()
            if approach_folder.is_dir()
        })




