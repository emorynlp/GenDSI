from __future__ import annotations
import typing as T

from matryoshki.data_view import DataView
from unique_names.generator import get_random_name
from matryoshki.serialize import save, load
from matryoshki.settings import Settings

import itertools
import pathlib
import inspect


F = T.TypeVar('F')


class Model:
    def __init__(self,
        submodel = None,
        root = None,
        pretrained = None,
    ):
        self.submodel = submodel
        self.root = pathlib.Path(root) if root else None
        for sup, sub in itertools.pairwise(self.chain):
            sub.root = sup.path_to_base
        existing_names = {
            folder.name for folder in self.path_to_model_type.iterdir() if folder.is_dir()
        } if self.path_to_model_type and self.path_to_model_type.exists() else set()
        self.name = get_random_name(existing_names)
        self.iteration = 0
        self._train_datas: list = [None]
        self.hyperparameters = {}
        pretrained_model_path = None
        pretrained_hyperparameters = {}
        final, final_i = None, -1
        if pretrained and pretrained.is_dir() and not pretrained.stem.startswith('_'):
            pretrained = pathlib.Path(pretrained)
            for subpath in pretrained.iterdir():
                if subpath.name[-1].isnumeric():
                    i = int(''.join((c for c in subpath.name if c.isnumeric())))
                    if i > final_i:
                        final, final_i = subpath, i
                elif subpath.name == 'best':
                    final = subpath
                    break
        for pretrained in ([pretrained, final] if final else [pretrained]):
            if pretrained and pretrained.is_dir():
                pretrained_hyperparameters_path = pretrained / 'hyperparameters.pkl'
                if pretrained_hyperparameters_path.exists():
                    pretrained_hyperparameters = load(pretrained_hyperparameters_path)
                    model_subpath = pretrained / 'model'
                    if model_subpath.exists():
                        pretrained_model_path = model_subpath
                    else:
                        glob_search = list(pretrained.glob('model.*'))
                        if glob_search:
                            pretrained_model_path = glob_search[0]
                    break
        if pretrained_model_path or pretrained_hyperparameters:
            self.hyperparameters = pretrained_hyperparameters
            self.hyperparameters['pretrained'] = str(pretrained_model_path)

    @classmethod
    def init(cls, __init__: F) -> F | callable:
        signature = inspect.signature(__init__)
        assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in signature.parameters.values())
        base_signature = inspect.signature(cls.__init__)
        def wrapped_init(*args, **kwargs):
            self = args[0]
            specified_args = signature.bind(*args, **kwargs).arguments
            all_args_binding = signature.bind(*args, **kwargs)
            all_args_binding.apply_defaults()
            all_args = all_args_binding.arguments
            cls.__init__(**{
                k: v for k, v in all_args.items() if k in base_signature.parameters
            })
            hyperparameters = all_args
            hyperparameters.update(self.hyperparameters)
            hyperparameters.update({
                k: v for k, v in specified_args.items()
                if k not in {'self', 'submodel', 'path'}
                and not k.startswith('_')
            })
            self.hyperparameters.update(hyperparameters)
            __init__(**hyperparameters)
            if self.path_to_hyperparameters is not None:
                save(self.hyperparameters, self.path_to_hyperparameters / 'hyperparameters.pkl')
            return
        return wrapped_init

    @property
    def chain(self):
        if not self.submodel:
            return [self]
        return [self] + self.submodel.levels

    @property
    def path_to_base(self):
        return self.root / self.__class__.__name__ if self.root else None

    @property
    def path_to_model_type(self):
        if not self.root:
            return None
        path = self.root
        for model in self.chain:
            cls_name = model.__class__.__name__
            path /= cls_name
        return path

    @property
    def path_to_preprocessed(self):
        return self.path_to_base / '_preprocessed' if self.path else None

    @property
    def path_to_postprocessed(self):
        return self.path_to_iteration / '_postprocessed' if self.path_to_iteration else None

    @property
    def path_to_predictions(self):
        return self.path_to_iteration / 'predictions' if self.path_to_iteration else None

    @property
    def path(self):
        return self.path_to_model_type / self.name if self.path_to_model_type else None

    @property
    def path_to_hyperparameters(self):
        return self.path / 'hyperparameters.pkl' if self.path else None

    @property
    def path_to_final_iteration(self):
        if self.path is None or not self.path.exists():
            return None
        folder = self.path / 'best'
        if folder.exists():
            return folder
        return self.path_to_last_iteration

    @property
    def path_to_final_eval(self):
        return self.path_to_final_iteration / 'eval' if self.path_to_final_iteration else None

    @property
    def path_to_final_model(self):
        return self.path_to_final_iteration / 'model' if self.path_to_final_iteration else None

    @property
    def path_to_last_iteration(self):
        if self.path is None or not self.path.exists():
            return None
        last = '0'
        last_i = -1
        for folder in self.path.iterdir():
            if folder.is_dir():
                if folder.name[-1].isnumeric():
                    i = int(''.join(filter(str.isnumeric, folder.name)))
                    if i > last_i:
                        last = folder
                        last_i = i
        return last

    @property
    def path_to_last_eval(self):
        return self.path_to_last_iteration / 'eval' if self.path_to_last_iteration else None

    @property
    def path_to_last_model(self):
        return self.path_to_last_iteration / 'model' if self.path_to_last_iteration else None

    @property
    def path_to_iteration(self):
        return self.path / str(self.iteration) if self.path else None

    @property
    def path_to_eval(self):
        return self.path_to_iteration / 'eval' if self.path_to_iteration else None

    @property
    def path_to_train(self):
        return self.path_to_iteration / 'train.json' if self.path_to_iteration else None

    @property
    def path_to_model(self):
        return self.path_to_iteration / 'model' if self.path_to_iteration else None

    @property
    def train_data(self):
        return self._train_datas[-1]

    @train_data.setter
    def train_data(self, value):
        self._train_datas[-1] = value

    def save(self, path: str |pathlib.Path = None):
        path = pathlib.Path(path) if path else self.path_to_model
        self._save(path)

    def _save(self, path):
        if self.submodel is not None:
            self.submodel.save(path)

    def preprocess(self, data, overwrite=False, **kwargs):
        path = get_path(data, folders=(self.root,))
        if path is not None and self.path_to_preprocessed is not None:
            preprocessed_data_path = self.path_to_preprocessed / f'{path.stem}.pkl'
            if not overwrite:
                preprocessed_data = get_data(preprocessed_data_path)
                if preprocessed_data is not None:
                    return preprocessed_data
        else:
            preprocessed_data_path = None
        preprocessed_data = self._preprocess(data)
        preprocessed_data = DataView(preprocessed_data, preprocessed_data_path)
        if preprocessed_data_path and preprocessed_data is not data:
            save(preprocessed_data, preprocessed_data_path)
        elif preprocessed_data_path and preprocessed_data is data:
            preprocessed_data_path.symlink_to(path)
        return preprocessed_data

    def _preprocess(self, data):
        return data

    def postprocess(self, data, overwrite=False):
        path = get_path(data, folders=(self.root,))
        if path is not None and self.path_to_postprocessed is not None:
            postprocessed_data_path = self.path_to_postprocessed / f'{path.stem}.pkl'
            if not overwrite:
                postprocessed_data = get_data(postprocessed_data_path)
                if postprocessed_data is not None:
                    return postprocessed_data
        else:
            postprocessed_data_path = None
        postprocessed_data = self._postprocess(data)
        postprocessed_data = DataView(postprocessed_data, postprocessed_data_path)
        if postprocessed_data_path and postprocessed_data is not data:
            save(postprocessed_data, postprocessed_data_path)
        elif postprocessed_data_path and postprocessed_data is data:
            postprocessed_data_path.symlink_to(path)
        return postprocessed_data

    def _postprocess(self, data):
        return data

    def predict(self, data, overwrite=False):
        preprocessed = self.preprocess(data)
        predictions_path = None
        if preprocessed.path is not None and self.path_to_predictions is not None:
            predictions_path = self.path_to_predictions / f'{preprocessed.path.stem}.pkl'
            if not overwrite:
                predictions = get_data(predictions_path)
                if predictions is not None:
                    return predictions
        predictions = self._predict(preprocessed)
        if predictions_path is not None:
            save(predictions, predictions_path)
        if predictions is None:
            predictions = self.submodel.predict(preprocessed)
        postprocessed = self.postprocess(predictions)
        return postprocessed

    def _predict(self, data):
        return None

    # def eval(self, data, predictions=None):
    #     all_results = {}
    #     preprocessed = self.preprocess(data)
    #     if predictions is None:
    #         if self.submodel:
    #             all_results = self.submodel.eval(preprocessed)
    #         predictions = self.predict(preprocessed)
    #     elif isinstance(predictions, (str, pathlib.Path)):
    #         predictions = self.postprocess(predictions)
    #     results = self._eval(preprocessed, predictions)
    #     all_results.update(results)
    #     path = preprocessed.path
    #     if path is not None and self.path_to_eval is not None:
    #         results_path = self.path_to_eval / self.name / f'{path.stem}.json'
    #         save(results, results_path)
    #     return results
    #
    # def _eval(self, data, predictions):
    #     return {}

    def train(self, *data):
        data = [DataView(d, folders=(self.root,)) for d in data]
        self._train_datas[-1] = data
        preprocessed = [self.preprocess(d) for d in data]
        training_generator = self._train(*preprocessed)
        submodel_is_training = False
        if training_generator is None:
            training_generator = self.submodel.training(*preprocessed)
            submodel_is_training = True
        path = preprocessed[0].path if preprocessed else None
        results = []
        if self.iteration == 0:
            self.iteration = 1
        for i, result in enumerate(training_generator):
            result['iteration'] = self.iteration
            self.iteration = i + 1
            results.append(result)
            if path is not None and not submodel_is_training and self.path_to_train is not None:
                results_path = self.path_to_train / f'{path.stem}.json'
                save(result, results_path)
            yield result
        self.hyperparameters['train_data'] = [
            str(d.path) for datas in self._train_datas if datas is not None for d in datas
        ]

    def _train(self, *data):
        return None



if __name__ == '__main__':
    print('Hello world')
