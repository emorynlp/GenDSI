

import ezpyz as ez
import dst.analysis.experiments_folder as exp
import tqdm
import pickle
import dst.data.dst_data as dst
import pathlib as pl
import dst.approaches.dst_seq_data as dstseq
from dst.approaches.seq2seq_dst import Seq2seqDst
from dst.approaches.seq2seq_dsg import Seq2seqDSG

Dialogue = dst.Dialogue


def clean_hyperparameter_saving_mistake():
    raise NotImplementedError # invalidated by change to experiments_folder.py
    for iteration in tqdm.tqdm(gather.iterations()): # noqa
        if iteration.hyperparameters.path.exists():
            hyperparameters = iteration.hyperparameters.load()
            for hyperparameter in hyperparameters:
                for hyperparam, hypervalue in list(hyperparameter.items()):
                    if not isinstance(hypervalue, str):
                        hyperparameter[hyperparam] = repr(hypervalue)
            iteration.hyperparameters.save(hyperparameters)
        dst_file = ez.Cache(iteration.model_path/'Seq2seqDst.pkl')
        s2s_file = ez.Cache(iteration.model_path/'Seq2seq.pkl')
        if dst_file.path.exists():
            dst = dst_file.load()
            hyperparameters = dst.hyperparameters
            for hyperparameter in hyperparameters:
                for hyperparam, hypervalue in list(hyperparameter.items()):
                    if not isinstance(hypervalue, str):
                        hyperparameter[hyperparam] = repr(hypervalue)
            dst_file.save(dst)
        if s2s_file.path.exists():
            s2s = s2s_file.load()
            hyperparameters = s2s.hyperparameters
            for hyperparameter in hyperparameters:
                for hyperparam, hypervalue in list(hyperparameter.items()):
                    if not isinstance(hypervalue, str):
                        hyperparameter[hyperparam] = repr(hypervalue)
            s2s_file.save(s2s)

def clean_data_mistake_predictions():
    experiments = exp.ExperimentsFolder()
    for approach in experiments.values():
        for file in tqdm.tqdm(approach.all_predictions()):
            predictions = file.load()
            if isinstance(predictions, dstseq.DstSeqData):
                fixed = dstseq.DstSeqData(predictions)
                file.save(fixed)

def clean_data_mistake_data():
    data_folder = pl.Path('data')
    for subdata in ('mwz2.4', 'gptdst5k'):
        subfolder = data_folder/subdata
        for file in tqdm.tqdm(list(subfolder.iterdir())):
            if file.is_file() and file.suffix == '.pkl':
                try:
                    data = ez.Cache(file).load()
                    if isinstance(data, dst.DstData):
                        fixed = dst.DstData(data)
                        ez.Cache(file).save(fixed)
                except Exception:
                    pass

import dst.approaches.seq2seq_dst as seq2seq_dst
def prompt_stub(*args, **kwargs):
    raise NotImplementedError
seq2seq_dst.prompt = prompt_stub
class PickleFixer(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "dst.approaches.seq2seq_dst"
        return super().find_class(module, name)
def clean_model_pickling_mistake():
    experiments = exp.ExperimentsFolder()
    for iteration in experiments['Seq2seqDst'].all_iterations():
        model_path = iteration.model_path/'Seq2seqDst.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as file:
                model = PickleFixer(file).load()
            if hasattr(model, 'prompt'):
                model.prompt = model.create_prompt
                with open(model_path, 'wb') as file:
                    pickle.dump(model, file)


def clean_example_turns_from_results():
    experiments = exp.ExperimentsFolder()
    for approach in experiments.values():
        for iteration in approach.all_iterations():
            for file in iteration.path.iterdir():
                if file.is_file() and file.stem != 'hyperparameters':
                    with open(file, 'rb') as f:
                        r = pickle.load(f)
                        if hasattr(r, 'good_slots'):
                            r.good_slots = None
                        if hasattr(r, 'bad_slots'):
                            r.bad_slots = None
                        if hasattr(r, 'good_turns'):
                            r.good_turns = None
                        if hasattr(r, 'bad_turns'):
                            r.bad_turns = None
                    with open(file, 'wb') as f:
                        pickle.dump(r, f)


def clean_hyperparameter_save_for_all_iterations():
    experiments = exp.ExperimentsFolder()
    for approach in experiments.values():
        for iteration in approach.all_iterations():
            model_path = iteration.model_path/'Seq2seqDst.pkl'
            if model_path.exists():
                model_obj = ez.File(model_path).load()
                hyperparameters = model_obj.hyperparameters
                ez.File(iteration.path/'hyperparameters.pkl').save(hyperparameters)

if __name__ == '__main__':
    clean_hyperparameter_save_for_all_iterations()