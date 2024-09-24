
from matryoshki.serialize import save, load

import pathlib
import pickle


class DataView:

    cache = {}

    @classmethod
    def cachegrab(cls, path, folder=None, *folders):
        original_path = path
        folders = (folder,) + folders if folder is not None else folders
        path = pathlib.Path(path)
        folders = tuple(pathlib.Path(folder) for folder in folders)
        if hasattr(cls.cache, '__contains__') and hasattr(cls.cache, '__getitem__'):
            if path in cls.cache:
                return cls.cache[path], path
            else:
                while True:
                    try:
                        data = load(path)
                        cls.cache[path] = data
                        break
                    except FileNotFoundError:
                        if folders and not path.is_absolute():
                            path = pathlib.Path(folders[0]) / path
                            folders = folders[1:]
                        else:
                            raise FileNotFoundError(
                                f'Could not find {original_path} in {folders}'
                            )
                return data, path
        else:
            while True:
                try:
                    return load(path), path
                except FileNotFoundError:
                    if folders and not path.is_absolute():
                        path = pathlib.Path(folders[0]) / path
                        folders = folders[1:]
                    else:
                        raise FileNotFoundError(
                            f'Could not find {original_path} in {folders}'
                        )

    def __new__(cls, raw=None, outputs=None, inputs=None, path=None, source=None, folder=None):
        """"""
        '''Support __new__ for pickle'''
        if all(arg is None for arg in (raw, outputs, inputs, path, source, folder)):
            return super().__new__(cls)
        '''Support calling like DataView(inputs, outputs), which sets raw to None'''
        if raw is not None and outputs is not None and inputs is None:
            raw, inputs, outputs = None, raw, outputs
        '''Any paths that are specified have data loaded and cached (or grabbed from cache)'''
        datas = [raw, inputs, outputs, source]
        paths = [path, None, None, None]
        for i, data in enumerate(datas):
            if isinstance(data, (str, pathlib.Path)):
                datas[i] = None
                paths[i] = data
            if isinstance(paths[i], (str, pathlib.Path)):
                paths[i] = pathlib.Path(paths[i])
                if all(x is None for x in datas[:3]):
                    datas[i], paths[i] = cls.cachegrab(paths[i], folder)
        raw, inputs, outputs, source = datas
        path, inputs_path, outputs_path, _ = paths
        '''DataView shallow copy constructor, resetting inputs and outputs views.
           DataView might be the object loaded from a path, in which case source may be a Path
           instead of loaded data.'''
        if isinstance(raw, DataView):
            path = path or raw.path  # noqa
            source = raw.source  # noqa
            inputs = False if raw.inputs in (False, None) else True
            outputs = False if raw.outputs in (False, None) else True
            raw = raw.raw  # noqa
        '''Constructing data from inputs and outputs if it is not specified.
           This shallow copies inputs and outputs rather than referencing them.'''
        if raw is None:
            if isinstance(inputs, DataView):
                inputs = inputs.pairs
            if isinstance(outputs, DataView):
                outputs = outputs.pairs
            if inputs not in (None, False) and outputs not in (None, False):
                raw = (list(inputs), list(outputs))
            elif inputs not in (None, False):
                raw = inputs
                outputs = False
                path = inputs_path
            elif outputs not in (None, False):
                raw = outputs
                inputs = False
                path = outputs_path
        '''Truish inputs/outputs become views and falsish inputs/outputs become None'''
        if inputs is False:
            inputs = None
        else:
            inputs = True
        if outputs is False:
            outputs = None
        else:
            outputs = True
        '''Prevent DataViews from being nested in DataViews by extracting raw data'''
        while isinstance(raw, DataView):
            raw = raw.raw
        '''Construct'''
        data = super().__new__(cls)
        data.__init__(raw, outputs, inputs, path, source, folder)
        return data

    def __init__(self, raw=None, outputs=True, inputs=True, path=None, source=None, folder=None):
        if hasattr(self, 'raw'):
            return
        self.raw = raw
        self.inputs = inputs
        self.outputs = outputs
        self.path = path
        self.source = source
        if self.inputs:
            self.inputs = InputsDataView(self) if outputs is not None else self
        if self.outputs:
            self.outputs = OutputsDataView(self) if inputs is not None else self
        self.pairs = PairsDataView(self)

    def save(self, path=None):
        if path is None:
            path = self.path
        copy = DataView(self)
        if isinstance(copy.source, DataView) and copy.source.path:
            copy.source = str(copy.source.path)
        elif isinstance(copy.source, (str, pathlib.Path)):
            copy.source = str(copy.source)
        else:
            copy.source = None
        copy.inputs = True if copy.inputs is not None else False
        copy.outputs = True if copy.outputs is not None else False
        save(copy, path)
        return self

    def __iter__(self):
        return iter(self.raw)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return DataView(self.raw[key], path=self.path, source=self.source)
        else:
            return self.raw[key]

    def __setitem__(self, key, value):
        self.raw[key] = value # noqa

    def __delitem__(self, key):
        del self.raw[key]

    def __len__(self):
        return len(self.raw)

    def __contains__(self, item):
        return item in self.raw

    def __str__(self):
        return f'DataView({str(self.raw)})'

    def __repr__(self):
        return f'Dataview({repr(self.raw)})'

    def append(self, item):
        return self.raw.append(item)

    def clear(self):
        return self.raw.clear()

    def count(self, item):
        return self.raw.count(item)

    def extend(self, iterable):
        return self.raw.extend(iterable)

    def index(self, item, start=None, end=None):
        return self.raw.index(item, start, end)

    def insert(self, index, item):
        return self.raw.insert(index, item)

    def pop(self, index=None):
        return self.raw.pop(index)

    def remove(self, item):
        return self.raw.remove(item)

    def reverse(self):
        return self.raw.reverse()

    def sort(self, key=None, reverse=False):
        return self.raw.sort(key=key, reverse=reverse)

    def update(self, other=None, **kwargs):
        return self.raw.update(other, **kwargs)

    def copy(self):
        return pickle.loads(pickle.dumps(self))



class PartialDataView:

    def __init__(self, data):
        self.data = data
        self.format = None

    @property
    def raw(self):
        return self.data.raw

    @property
    def inputs(self):
        return self.data.inputs

    @property
    def outputs(self):
        return self.data.outputs

    @property
    def path(self):
        return self.data.path

    @property
    def source(self):
        return self.data.source

    def __iter__(self):
        return iter(self.format)

    def __getitem__(self, key):
        return self.format[key]

    def __setitem__(self, key, value):
        self.format[key] = value  # noqa

    def __delitem__(self, key):
        del self.format[key]

    def __len__(self):
        return len(self.format)

    def __contains__(self, item):
        return item in self.format

    def __str__(self):
        return str(self.format)

    def __repr__(self):
        return repr(self.format)

    def append(self, item):
        return self.format.append(item)

    def clear(self):
        return self.format.clear()

    def count(self, item):
        return self.format.count(item)

    def extend(self, iterable):
        return self.format.extend(iterable)

    def index(self, item, start=None, end=None):
        return self.format.index(item, start, end)

    def insert(self, index, item):
        return self.format.insert(index, item)

    def pop(self, index=None):
        return self.format.pop(index)

    def remove(self, item):
        return self.format.remove(item)

    def reverse(self):
        return self.format.reverse()

    def sort(self, key=None, reverse=False):
        return self.format.sort(key=key, reverse=reverse)

    def update(self, other=None, **kwargs):
        return self.format.update(other, **kwargs)


class PairsDataView(PartialDataView):
    def __init__(self, data):
        super().__init__(data)
        if data.inputs and data.outputs:
            self.format = (data.inputs, data.outputs)
        elif data.inputs:
            self.format = (data.inputs, (None,) * len(data.inputs))
        elif data.outputs:
            self.format = ((None,) * len(data.outputs), data.outputs)

    @property
    def inputs(self):
        return self.format[0]

    @property
    def outputs(self):
        return self.format[1]

    def copy(self):
        return pickle.loads(pickle.dumps(self.data)).pairs

    def __iter__(self):
        return zip(self.inputs, self.outputs)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.inputs[key], self.outputs[key]
        elif isinstance(key, slice):
            inputs = self.inputs[key]
            outputs = self.outputs[key]
            return self.__class__(self.data.__class__(inputs, outputs))
        else:
            raise TypeError("Invalid key type. Must be int or slice.")

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.inputs[key], self.outputs[key] = value
        elif isinstance(key, slice):
            if isinstance(value, PairsDataView):
                self.inputs[key] = value.inputs
                self.outputs[key] = value.outputs
            elif isinstance(value, tuple):
                inputs, outputs = value
                self.inputs[key] = inputs
                self.outputs[key] = outputs
            else:
                raise TypeError("Invalid value type. Must be PairsDataView or tuple.")
        else:
            raise TypeError("Invalid key type. Must be int or slice.")

    def __delitem__(self, key):
        if isinstance(key, int):
            del self.inputs[key]
            del self.outputs[key]
        elif isinstance(key, slice):
            del self.inputs[key]
            del self.outputs[key]
        else:
            raise TypeError("Invalid key type. Must be int or slice.")

    def __len__(self):
        return len(self.inputs)

    def __contains__(self, item):
        return item in self.inputs or item in self.outputs

    def __str__(self):
        return f"PairsDataView(inputs={self.inputs}, outputs={self.outputs})"

    def __repr__(self):
        return f"PairsDataView(inputs={self.inputs}, outputs={self.outputs})"

    def append(self, item):
        inputs, outputs = item
        self.inputs.append(inputs)
        self.outputs.append(outputs)

    def clear(self):
        self.inputs.clear()
        self.outputs.clear()

    def count(self, item):
        return self.inputs.count(item)

    def extend(self, iterable):
        if isinstance(iterable, PairsDataView):
            self.inputs.extend(iterable.inputs)
            self.outputs.extend(iterable.outputs)
        else:
            inputs, outputs = zip(*iterable)
            self.inputs.extend(inputs)
            self.outputs.extend(outputs)

    def index(self, item, start=None, end=None):
        if start is None:
            start = 0
        if end is None:
            end = len(self.inputs)
        index_inputs = self.inputs.index(item[0], start, end)
        index_outputs = self.outputs.index(item[1], start, end)
        if index_inputs == index_outputs:
            return index_inputs
        else:
            raise ValueError(f"{item} is not in PairsDataView")

    def insert(self, index, item):
        inputs, outputs = item
        self.inputs.insert(index, inputs)
        self.outputs.insert(index, outputs)

    def pop(self, index=None):
        inputs = self.inputs.pop(index)
        outputs = self.outputs.pop(index)
        return inputs, outputs

    def remove(self, item):
        inputs, outputs = item
        self.inputs.remove(inputs)
        self.outputs.remove(outputs)

    def reverse(self):
        self.inputs.reverse()
        self.outputs.reverse()

    def sort(self, key=None, reverse=False):
        if key is None:
            key = lambda pair: pair[0]
        pairs = list(zip(self.inputs, self.outputs))
        pairs.sort(key=key, reverse=reverse)
        for i, (input, output) in enumerate(pairs):
            self.inputs[i] = input
            self.outputs[i] = output

    def update(self, other=None, **kwargs):
        if other is not None:
            if isinstance(other, PairsDataView):
                self.inputs.extend(other.inputs)
                self.outputs.extend(other.outputs)
            else:
                inputs, outputs = zip(*other)
                self.inputs.extend(inputs)
                self.outputs.extend(outputs)
        if kwargs:
            inputs, outputs = zip(*kwargs.items())
            self.inputs.extend(inputs)
            self.outputs.extend(outputs)


class InputsDataView(PartialDataView):
    def __init__(self, data):
        super().__init__(data)
        if isinstance(data.raw, tuple) and len(data.raw) == 2:
            self.format = InputsOutputsTuple(data.raw[0])
        elif (
            hasattr(data.raw, '__len__') and len(data.raw) > 0
            and isinstance(data.raw[0], dict) and 'input' in data.raw[0]
        ):
            self.format = DictsWithInputOutputKeys(data.raw, 'input')
        else:
            self.format = InputOutputPairs(data.raw, 0)

    def copy(self):
        return pickle.loads(pickle.dumps(self.data)).inputs

class OutputsDataView(PartialDataView):
    def __init__(self, data):
        super().__init__(data)
        if isinstance(data.raw, tuple) and len(data.raw) == 2:
            self.format = InputsOutputsTuple(data.raw[1])
        elif (
            hasattr(data.raw, '__len__') and len(data.raw) > 0
            and isinstance(data.raw[0], dict) and 'output' in data.raw[0]
        ):
            self.format = DictsWithInputOutputKeys(data.raw, 'output')
        else:
            self.format = InputOutputPairs(data.raw, 1)

    def copy(self):
        return pickle.loads(pickle.dumps(self.data)).outputs


class InputsOutputsTuple:

    def __init__(self, raw):
        self.raw = raw

    def __iter__(self):
        return iter(self.raw)

    def __getitem__(self, key):
        return self.raw[key]

    def __setitem__(self, key, value):
        self.raw[key] = value # noqa

    def __delitem__(self, key):
        del self.raw[key]

    def __len__(self):
        return len(self.raw)

    def __contains__(self, item):
        return item in self.raw

    def __str__(self):
        return str(self.raw)

    def __repr__(self):
        return repr(self.raw)

    def append(self, item):
        return self.raw.append(item)

    def clear(self):
        return self.raw.clear()

    def count(self, item):
        return self.raw.count(item)

    def extend(self, iterable):
        return self.raw.extend(iterable)

    def index(self, item, start=None, end=None):
        return self.raw.index(item, start, end)

    def insert(self, index, item):
        return self.raw.insert(index, item)

    def pop(self, index=None):
        return self.raw.pop(index)

    def remove(self, item):
        return self.raw.remove(item)

    def reverse(self):
        return self.raw.reverse()

    def sort(self, key=None, reverse=False):
        return self.raw.sort(key=key, reverse=reverse)

    def update(self, other=None, **kwargs):
        return self.raw.update(other, **kwargs)


class DictsWithInputOutputKeys:
    def __init__(self, raw, key):
        self.raw = raw
        self.key = key

    def __iter__(self):
        return (item[self.key] for item in self.raw)

    def __getitem__(self, index):
        return self.raw[index][self.key]

    def __setitem__(self, index, value):
        self.raw[index][self.key] = value

    def __delitem__(self, index):
        del self.raw[index][self.key]

    def __len__(self):
        return len(self.raw)

    def __contains__(self, value):
        return any(value == item[self.key] for item in self.raw)

    def __str__(self):
        return str([item[self.key] for item in self.raw])

    def __repr__(self):
        return repr([item[self.key] for item in self.raw])

    def append(self, value):
        self.raw.append({self.key: value})

    def clear(self):
        for item in self.raw:
            del item[self.key]

    def count(self, value):
        return sum(value == item[self.key] for item in self.raw)

    def extend(self, iterable):
        for value in iterable:
            self.append(value)

    def index(self, value, start=None, end=None):
        start = start or 0
        end = end or len(self.raw)
        for i in range(start, end):
            if self.raw[i][self.key] == value:
                return i
        raise ValueError(f"{value} is not in list.")

    def insert(self, index, value):
        self.raw.insert(index, {self.key: value})

    def pop(self, index=None):
        if index is None:
            return self.raw.pop()[self.key]
        return self.raw.pop(index)[self.key]

    def remove(self, value):
        for item in self.raw:
            if item[self.key] == value:
                self.raw.remove(item)
                return
        raise ValueError(f"{value} is not in list.")

    def reverse(self):
        self.raw.reverse()

    def sort(self, key=None, reverse=False):
        self.raw.sort(
            key=lambda item: item[self.key] if key is None else key(item[self.key]),
            reverse=reverse
        )

    def update(self, other=None, **kwargs):
        if other is None:
            other = kwargs
        for item in other:
            if self.key in item:
                index = self.index(item[self.key])
                self.raw[index].update(item)
            else:
                self.append(item[self.key])


class InputOutputPairs:
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __iter__(self):
        return (item[self.index] for item in self.data)

    def __getitem__(self, idx):
        return self.data[idx][self.index]

    def __setitem__(self, idx, value):
        self.data[idx] = (self.data[idx][0], value)

    def __delitem__(self, idx):
        del self.data[idx]

    def __len__(self):
        return len(self.data)

    def __contains__(self, value):
        return any(value == item[self.index] for item in self.data)

    def __str__(self):
        return str([item[self.index] for item in self.data])

    def __repr__(self):
        return repr([item[self.index] for item in self.data])

    def append(self, value):
        self.data.append((None, value))

    def clear(self):
        self.data.clear()

    def count(self, value):
        return sum(value == item[self.index] for item in self.data)

    def extend(self, iterable):
        self.data.extend((None, value) for value in iterable)

    def index(self, value, start=None, end=None):
        start = start or 0
        end = end or len(self.data)
        for i in range(start, end):
            if self.data[i][self.index] == value:
                return i
        raise ValueError(f"{value} is not in list.")

    def insert(self, index, value):
        self.data.insert(index, (None, value))

    def pop(self, index=None):
        if index is None:
            return self.data.pop()[self.index]
        return self.data.pop(index)[self.index]

    def remove(self, value):
        for item in self.data:
            if item[self.index] == value:
                self.data.remove(item)
                return
        raise ValueError(f"{value} is not in list.")

    def reverse(self):
        self.data.reverse()

    def sort(self, key=None, reverse=False):
        self.data.sort(
            key=lambda item: item[self.index] if key is None else key(item[self.index]),
            reverse=reverse
        )

    def update(self, other=None, **kwargs):
        if other is None:
            other = kwargs
        for item in other:
            if self.index in item:
                index = self.index(item[self.index])
                self.data[index] = (self.data[index][0], item[self.index])
            else:
                self.append(item[self.index])


