
import ezpyz as ez
from dataclasses import dataclass


@dataclass
class Results(ez.Data):

    def update(self, *datas):
        no_metric = object()
        for data in datas:
            for key, value in vars(self).items():
                if value is None and hasattr(data, key):
                    metric_fn = getattr(data, key, no_metric)
                    if metric_fn is not no_metric and callable(metric_fn):
                        metric_fn(self)
        return self

    def display(self):
        return f'{type(self).__name__}:\n' + '\n'.join(
            f"    {key}: {value}"
            for key, value in vars(self).items()
            if value is not None and value is not ...
        )