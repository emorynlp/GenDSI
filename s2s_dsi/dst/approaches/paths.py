

import ezpyz as ez

import pathlib


class Paths:

    def __init__(self,
        task:ez.filelike=None,
        approach=None,
        experiment:str=None,
        iteration: str | int=0
    ):
        self.task:pathlib.Path = ez.File(task).path
        self.approach_name:str = approach or '.'
        self.experiment_name:str = experiment or ez.denominate()
        self.iteration_name:str = str(iteration) or '0'
    update = __init__

    def approach(self):
        return self.task / self.approach_name

    def experiment(self):
        return self.approach() / self.experiment_name

    def iteration(self):
        return self.experiment() / self.iteration_name

    def model(self, *parts):
        return ez.File(pathlib.Path(self.iteration(), 'model', *parts))

    def results(self, *parts):
        return ez.File(pathlib.Path(self.iteration(), 'results', *parts))

    def predictions(self, *parts):
        return ez.File(pathlib.Path(self.iteration(), 'predictions', *parts))



