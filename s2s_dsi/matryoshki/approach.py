from __future__ import annotations

import typing as T

from matryoshki.data_view import DataView
from matryoshki.unique_names.generator import get_random_name
from matryoshki.serialize import save, load
from matryoshki.settings import settings, SettingsDict

import copy
import pathlib
import inspect
import itertools
import pickle


F = T.TypeVar('F')
C = T.TypeVar('C')


class Approach:
    def __init__(self,
        path_to_task: str | pathlib.Path = None,
        subapproach: Approach = None,
        superapproach: Approach = None,
        experiment_name: str = None,
        hyperparameters: settings = None,
    ):
        if isinstance(path_to_task, str):
            path_to_task = pathlib.Path(path_to_task)
        self.path_to_task = path_to_task
        self.path = Paths(self)
        self.subapproach = subapproach
        self.superapproach = superapproach
        self.experiment_name = experiment_name or get_random_name({
            folder.name for folder in self.path.approach.iterdir()
        } if self.path.approach and self.path.approach.exists() else None)
        for level in self.sublevels:
            if isinstance(level, Approach):
                level.path_to_task = self.path_to_task
                level.experiment_name = self.experiment_name
        if isinstance(self.subapproach, Approach):
            self.subapproach.superapproach = self
        self.hyperparameters = hyperparameters
        self.iteration = 0

    @classmethod
    def init(cls: C, __init__) -> T.Callable[..., C]:
        def settings_catcher(self, *args, settings=None, **kwargs):
            __init__(self, *args, **kwargs)
            self.hyperparameters = Hyperparameters(self, settings)
            return
        super_signature = inspect.signature(cls.__init__)
        sub_signature = inspect.signature(__init__)
        settings_init = settings(sub_signature, self=False)(settings_catcher)
        def wrapper(self, *args, **kwargs):
            super_kwargs = {}
            binding = sub_signature.bind(self, *args, **kwargs)
            binding.apply_defaults()
            for name, parameter in super_signature.parameters.items():
                if name in binding.arguments:
                    super_kwargs[name] = binding.arguments[name]
            cls.__init__(**super_kwargs)
            settings_init(self, *args, **kwargs)
        return wrapper

    def preprocess(self, data=None, outputs=None, inputs=None):
        data = DataView(data, outputs=outputs, inputs=inputs, folder=self.path_to_task)
        if self.path.preprocessed and data.path:
            path = self.path.preprocessed / f'{data.path.stem}.pkl'
        else:
            path = None
        if path and path.exists():
            return DataView(path)
        try:
            preprocessed = self._preprocess(data)
        except NotImplementedError:
            return data
        preprocessed = DataView(preprocessed, path=path, source=data)
        return preprocessed

    def _preprocess(self, data):
        raise NotImplementedError

    def postprocess(self, data=None, outputs=None, inputs=None):
        data = DataView(data, outputs=outputs, inputs=inputs, folder=self.path_to_task)
        if self.path.postprocessed and data.path:
            path = self.path.postprocessed / f'{data.path.stem}.pkl'
        else:
            path = None
        try:
            postprocessed = self._postprocess(data)
        except NotImplementedError:
            return data
        postprocessed = DataView(postprocessed, path=path, source=data)
        return postprocessed

    def _postprocess(self, data):
        raise NotImplementedError

    def predict(self, data=None, outputs=None, inputs=None):
        data = DataView(data, outputs=outputs, inputs=inputs, folder=self.path_to_task)
        preprocessed = self.preprocess(data)
        if self.path.predictions and data.path:
            path = self.path.predictions / f'{data.path.stem}.pkl'
        else:
            path = None
        predictions = self._predict(preprocessed)
        predictions = DataView(preprocessed.inputs, predictions, path=path, source=data)
        predictions = self.postprocess(predictions)
        predictions.path = path
        return predictions

    def __call__(self, data=None, outputs=None, inputs=None):
        return self.predict(data, outputs, inputs)

    def _predict(self, data):
        return self.subapproach.predict(data).outputs

    def eval(self, data=None, outputs=None, inputs=None, predictions=None, **metrics):
        if data in (True, False, ..., 'no', 'none'):
            data = None
        elif data is not None:
            data = DataView(data, outputs=outputs, inputs=inputs, folder=self.path_to_task)
        if predictions is not None:
            if predictions not in (True, False, ..., 'no', 'none'):
                predictions = DataView(predictions)
            else:
                predictions = None
            if predictions is not None and predictions.path and self.path.eval:
                path = self.path.eval / f'{predictions.path.stem}.json'
            elif data is not None and data.path and self.path.eval:
                path = self.path.eval / f'{data.path.stem}.json'
            else:
                path = None
            all_metrics = {}
            all_metrics.update(self._eval(data, predictions))
            for metric_name, metric in metrics.items():
                if callable(metric):
                    all_metrics[metric_name] = metric(data, predictions)
                else:
                    all_metrics[metric_name] = metric
            datasource = dict(
                data=str(data.path),
            ) if isinstance(data, DataView) and data.path and data.path.exists() else {}
            predictionsource = (dict(
                predictions=str(predictions.path),
            ) if
                isinstance(predictions, DataView) and
                predictions.path and predictions.path.exists() else {}
            )
            all_metrics = EvalResults(
                self, path, all_metrics, **datasource, **predictionsource
            )
            return all_metrics
        else:
            predictions = self.predict(data)
            return self.eval(data, predictions=predictions, **metrics)


    def _eval(self, data, predictions):
        if self.subapproach is None:
            return {}
        else:
            return self.subapproach.eval(data, predictions)

    def training(self, *data, resume=False):
        if not resume:
            self.iteration = 0
        data = [self.preprocess(d) for d in data]
        training = self._training(*data)
        while True:
            self.iteration += 1
            try:
                result = next(training)
                result = TrainingResults(self, result)
                yield result
            except StopIteration:
                break

    def _training(self, *data):
        for result in self.subapproach.training(*data):
            yield result

    def train(self, *data):
        return list(self.training(*data))

    def save(self, path=None):
        if path is None:
            path = self.path.model
        chain = []
        for submodel, supermodel in itertools.pairwise(reversed([self]+self.sublevels)):
            submodel = copy.copy(submodel)
            chain.append(submodel)
        savable = copy.copy(self)
        chain.append(savable)
        for submodel, supermodel in itertools.pairwise(chain):
            submodel.superapproach = supermodel
            supermodel.subapproach = submodel
            submodel.path = Paths(submodel)
            submodel._save(path)
        savable.superapproach = None
        savable.path = Paths(savable)
        path.mkdir(parents=True, exist_ok=True)
        result = savable._save(path) # noqa
        if result is not None:
            savable = result
        save(savable, path/'model.pkl')
        return self

    def _save(self, path):
        return

    @classmethod
    def load(cls, path):
        path = pathlib.Path(path)
        if path.name.isnumeric():
            path = path / 'model'
        elif path.suffix != 'model':
            highest_iteration = 0
            if path.exists():
                for folder in path.iterdir():
                    if folder.is_dir():
                        if folder.name[-1].isnumeric():
                            i = int(''.join(filter(str.isnumeric, folder.name)))
                            if i > highest_iteration:
                                highest_iteration = i
            path = path / str(highest_iteration) / 'model'
        obj_path = path / 'model.pkl'
        with open(obj_path, 'rb') as file:
            self = pickle.load(file)
        for model in reversed(self.levels):
            model._load(path)
        return self

    def _load(self, path):
        return self

    @property
    def approach_name(self):
        return self.__class__.__name__

    @property
    def levels(self):
        return self.superlevels + [self] + self.sublevels

    @property
    def sublevels(self):
        models = [self]
        while models[-1].subapproach:
            models.append(models[-1].subapproach)
        return models[1:]

    @property
    def superlevels(self):
        models = [self]
        while models[-1].superapproach:
            models.append(models[-1].superapproach)
        models.reverse()
        return models[:-1]



class EvalResults(dict):

    def __init__(self, model: Approach, path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.path = path

    def save(self, path=None):
        if path is None:
            path = self.path
        save(self, path)
        if self.model.path.experiment and not self.model.path.hyperparameters.exists():
            if isinstance(self.model.hyperparameters, Hyperparameters):
                self.model.hyperparameters.save()
        return self


class TrainingResults(dict):

        def __init__(self, model: Approach, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = model

        def save(self, path=None):
            if path is None:
                path = self.model.path.train_metrics
            save(self, path, append=True)
            if self.model.path.experiment and not self.model.path.hyperparameters.exists():
                if isinstance(self.model.hyperparameters, Hyperparameters):
                    self.model.hyperparameters.save()
            return self


class Hyperparameters(SettingsDict):

    def __init__(self, model: Approach, *args, **kwargs):
        kwa = dict(
            approach = model.approach_name,
            subapproach = model.subapproach.hyperparameters if model.subapproach else None,
            path_to_experiment = str(model.path.experiment),
        )
        kwa.update(kwargs)
        super().__init__(*args, **kwa)
        self.specified = None
        self.defaults = None
        self.args = None
        self.kwargs = None

    def save(self, path=None):
        if path is None:
            path = self.get('path_to_experiment')
            if path is not None:
                path = pathlib.Path(path) / 'hyperparameters.pkl'
        save(self, path)
        return self


class Paths:
    """
    Defines a matryoshki directory structure.

    If an Approach obj is given, it will look at the obj (it's class name, experiment name, iteration, etc.) to define directory structure paths.

    Alternatively, the names of the task, approach, experiment, etc. can be specified in the Path object (without an Approach obj) to define the directory structure.
    """

    def __init__(self,
        obj: Approach = None,
        task = None,
        superapproaches = None,
        approach=None,
        subapproaches = None,
        experiment = None,
        iteration = None,
    ):
        self._obj: Approach = obj
        self._task = task
        self._experiment = experiment
        self._superapproaches = superapproaches
        self._approach = approach
        self._subapproaches = subapproaches
        self._iteration = iteration

    @property
    def _i(self):
        return self._obj.iteration if self._obj else self._iteration

    @property
    def _t(self):
        return self._obj.path_to_task if self._obj else self._task

    @property
    def _e(self):
        return self._obj.experiment_name if self._obj else self._experiment

    @property
    def _a(self):
        return self._obj.approach_name if self._obj else self._approach

    @property
    def _sub(self):
        if self._obj:
            return [model.approach_name for model in self._obj.sublevels]
        else:
            return self._subapproaches

    @property
    def _super(self):
        if self._obj:
            return [model.approach_name for model in self._obj.superlevels]
        else:
            return self._superapproaches

    @property
    def task(self):
        """The root folder representing what task is being modeled. Has raw data and Approaches"""
        return self._t

    @property
    def approach(self):
        """Path under task representing an approach chain (points to the end of the chain)"""
        return pathlib.Path(self.task, *self._super, self._a) if self.task else None

    @property
    def preprocessed(self):
        """Path to where preprocessed data is saved and loaded, under the approach folder"""
        return self.approach / '_preprocessed' if self.task else None

    @property
    def postprocessed(self):
        """Path to where postprocessed data is saved and loaded, under the approach folder"""
        return self.approach / '_postprocessed' if self.task else None

    @property
    def experiment(self) -> pathlib.Path | None:
        """Path under the approach folder containing data from a single experiment (run of the code)"""
        if self.approach and self._e:
            return pathlib.Path(self.approach, *self._sub, self._e)
        else:
            return None

    @property
    def hyperparameters(self):
        """Pickle file of hyperparameters (arguments passed to approach object) to run the experiment"""
        return self.experiment / 'hyperparameters.pkl' if self.experiment else None

    @property
    def last_iteration(self):
        """Path to the folder containing data for the model during/after the last iteration of training """
        if not self.experiment:
            return None
        highest_iteration = 0
        if self.experiment and self.experiment.exists():
            for folder in self.experiment.iterdir():
                if folder.is_dir():
                    if folder.name[-1].isnumeric():
                        i = int(''.join(filter(str.isnumeric, folder.name)))
                        if i > highest_iteration:
                            highest_iteration = i
        return self.experiment / str(highest_iteration)

    @property
    def best_iteration(self):
        """Path to the folder containing data for the model during/after the best iteration of training"""
        if self.experiment is None:
            return None
        return self.experiment / 'best'

    @property
    def final_iteration(self):
        """Path to the folder containing data for the model during/after the best (or last) iteration of training"""
        return self.best_iteration or self.last_iteration

    @property
    def iteration(self):
        """Path to the folder containing data for the model during/after the current iteration of training"""
        if self._i is not None:
            return self.experiment / str(self._i) if self.experiment else None
        else:
            return None

    @property
    def train_metrics(self):
        """Path to the file containing training metrics for the current iteration of training"""
        return self.iteration / 'train.json' if self.iteration else None

    @property
    def eval(self):
        """Path to the folder containing evaluation metrics for the current model iteration"""
        return self.iteration / 'eval' if self.iteration else None

    @property
    def predictions(self):
        """Path to the folder containing predictions for the current model iteration"""
        return self.iteration / 'predictions' if self.iteration else None

    @property
    def model(self):
        """Path to the folder containing the saved model, which can be loaded in future experiments"""
        return self.iteration / 'model' if self.iteration else None





if __name__ == '__main__':
    print('Hello world')
