
import pathlib

ConcretePathType = type(pathlib.Path())


class Path(ConcretePathType):

    def __new__(cls, approach=None, experiment=None, iteration=None, task=None):
        approach = pathlib.Path(approach) if approach else None
        experiment = pathlib.Path(experiment) if experiment else None
        iteration = pathlib.Path(str(iteration)) if iteration else None
        task = pathlib.Path(task) if task else None
        if iteration and not experiment:
            experiment = iteration.parent
            iteration = iteration.name
        if experiment and not approach:
            approach = experiment.parent
            experiment = experiment.name
        if approach and not task:
            if approach.is_relative_to(pathlib.Path.cwd()):
                relative = task.relative_to(pathlib.Path.cwd())
                if len(relative.parts) > 1:
                    task = relative.parts[0]
                else:
                    task = pathlib.Path.cwd()
        if task and approach:
            approach = task / approach
        if approach and experiment:
            assert len(experiment.parts) == 1
            experiment = approach / experiment
        if experiment and iteration:
            assert len(iteration.parts) == 1
            iteration = experiment / iteration

        path = iteration or experiment or approach or task
        if path:
            return super().__new__(cls, path)
        else:
            return super().__new__(cls)




if __name__ == '__main__':

    path = Path('task/Foo/Bar/furious_yoda/2')
    print(path.task)
    print(path.approach)
    print(path)
    print(type(path))
