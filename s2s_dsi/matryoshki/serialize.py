
import pathlib
import csv
import json
import pickle


def save(result, path, append=False):
    save_path = pathlib.Path(path)
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True)
    if save_path.suffix == '.csv':
        result = builtins_to_json_serializable(result)
        with open(save_path, 'a' if append else 'w') as file:
            writer = csv.writer(file)
            writer.writerows(result)
    elif save_path.suffix == '.json':
        result = builtins_to_json_serializable(result)
        with open(save_path, 'w') as file:
            json.dump(result, file)
    elif save_path.suffix == '.jsonl':
        result = builtins_to_json_serializable(result)
        with open(save_path, 'a' if append else 'w') as file:
            file.write(json.dumps(result) + '\n')
    else:
        with open(save_path, 'wb') as file:
            pickle.dump(result, file)

def load(path):
    load_path = pathlib.Path(path)
    if load_path.suffix == '.csv':
        with open(load_path, 'r') as file:
            reader = csv.reader(file)
            loaded = list(reader)
    elif load_path.suffix == '.json':
        with open(load_path, 'r') as file:
            loaded = json.load(file)
    else:
        with open(load_path, 'rb') as file:
            loaded = pickle.load(file)
    return loaded

def builtins_to_json_serializable(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, list):
        return [builtins_to_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {builtins_to_json_serializable(key): builtins_to_json_serializable(value) for key, value in obj.items()}
    if hasattr(obj, '__iter__'):
        return [builtins_to_json_serializable(x) for x in obj]
    return str(obj)