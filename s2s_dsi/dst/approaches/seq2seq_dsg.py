import copy
import itertools
import pathlib
import fire
import traceback
import ezpyz as ez
from dst.approaches.dst_seq_data import DstSeqData, DstSeqResults
from dst.data.dst_data import Slot
from dst.approaches.sequences_data import SequencesData, SequenceResults, Sequence
from dst.approaches.sequences_data import Hyperparameters
import typing as T
class NoException(BaseException): pass


class Seq2SeqModel(T.Protocol):
    @property
    def hyperparameters(self) -> Hyperparameters: return ...
    def predict(self, data:SequencesData) -> SequencesData: ...
    def logits(self, data:SequencesData) -> SequencesData: ...
    def training(self, data:SequencesData) -> T.Iterable[SequenceResults]: ...
    def train(self, data:SequencesData) -> list[SequenceResults]: ...
    def save(self, path:ez.filelike): ...
    @classmethod
    def load(cls, file:ez.filelike, directory=None, device=None) -> 'Seq2SeqModel': ...



class Seq2seqDSG:
    def __init__(self,
        seq2seq_model:Seq2SeqModel,
        directory=None,
        approach=None,
        experiment=None,
        max_context_turns=None,
        sep_token = '**',
        eoi_token = '->',
        req_token = '?',
    ):
        self.seq2seq_model:Seq2SeqModel = seq2seq_model
        self.directory = pathlib.Path('.' if directory is None else directory)
        self.approach = approach or type(self).__name__
        self.experiment = experiment or ez.denominate({
            experiment.name for experiment in (
                (self.directory / self.approach).iterdir()
                if (self.directory / self.approach).exists() else ()
            )
        })
        self.iteration = 0
        self.hyperparameters = Hyperparameters(
            pathlib.Path(
                self.directory, self.approach, self.experiment, str(self.iteration),
                f"hyperparameters.pkl"
            )
        )
        self.record_hyperparameters('init')
        self.max_context_turns = max_context_turns
        self.sep_token = sep_token
        self.eoi_token = eoi_token
        self.req_token = req_token

    def save(self, path:ez.filelike=None):
        if path is None:
            path = pathlib.Path(self.path_prefix(), 'model')
        path = ez.File(path).path
        self.seq2seq_model.save(path)
        scopy = copy.copy(self)
        scopy.seq2seq_model = None
        ez.Cache(path/'Seq2seqDst.pkl').save(scopy)

    @classmethod
    def load(cls, path:ez.filelike, directory=None, device=None, load_in_8bit=None) -> 'Seq2seqDSG':
        path = ez.File(path).path
        seq2seq_dst = ez.Cache(path/'Seq2seqDst.pkl').load()
        seq2seq = ez.Cache(path/'Seq2seq.pkl').load()
        seq2seq = seq2seq.load(path=path, directory=directory, device=device, load_in_8bit=load_in_8bit)
        seq2seq_dst.seq2seq_model = seq2seq
        return seq2seq_dst

    def record_hyperparameters(self, stage, **kwargs):
        self.hyperparameters.record(self, stage=stage, **kwargs)
        self.hyperparameters[-1] = {
            **self.seq2seq_model.hyperparameters[-1], **self.hyperparameters[-1]
        }

    def path_prefix(self):
        return pathlib.Path(self.directory) / self.approach / self.experiment / str(self.iteration)

    def preprocess(self, data:DstSeqData) -> DstSeqData:
        for dialogue in data.dialogues:
            for turn in [turn for turn in dialogue.turns]:
                turn.state_sequence = Sequence()
                context = turn.context()
                if self.max_context_turns is not None:
                    context = context[-self.max_context_turns:]
                seq_in = ''
                for i, (speaker, context_turn) in enumerate(reversed(
                    list(zip(itertools.cycle('AB'), context))
                )):
                    if i == len(context) - 1:
                        seq_in = f"** {speaker}: {context_turn.turn}\n" + seq_in
                    else:
                        seq_in = f"{speaker}: {context_turn.turn}\n"+ seq_in
                seq_in += self.eoi_token
                turn.state_sequence.seq_input = seq_in
                seq_label = ''
                for slot, values in (turn.slots or {}).items():
                    if values is not None:
                        if '?' in values:
                            values = [self.req_token if v == '?' else v for v in values]
                        if values:
                            seq_label += f"{slot.name}: {', '.join(values)} {self.sep_token}\n"
                turn.state_sequence.seq_label = seq_label
        if data.file is not None:
            data.file = self.path_prefix().parent / 'preprocess' /\
                        f"{data.file.path.stem}{data.file.format.extensions[0]}"
        return data

    def postprocess(self, data:DstSeqData):
        for dialogue in data.dialogues:
            for turn in (
                turn for turn in dialogue.turns
                if turn.state_sequence and turn.state_sequence.seq_output is not None
            ):
                turn.predicted_slots = {}
                seq_out = turn.state_sequence.seq_output
                for line in seq_out.split(self.sep_token):
                    if ':' in line:
                        slot_name, value = line.split(':', 1)
                        values = [v.strip() for v in value.split(',')]
                        values = ['?' if v == self.req_token else v for v in values if v]
                        if slot_name:
                            slot = Slot(slot_name.strip())
                            turn.predicted_slots[slot] = values
        if data.file is not None:
            data.file = self.path_prefix().parent / 'postprocess' /\
                        f"{data.file.path.stem}{data.file.format.extensions[0]}"
        return data

    def predict(self, data:DstSeqData):
        from_file = data.file
        data = self.preprocess(data)
        sequences = data.state_sequences()
        predicted_sequences = self.seq2seq_model.predict(sequences)
        data = self.postprocess(data)
        if from_file is not None:
            data.file = self.path_prefix() / 'predictions' /\
                        f"{from_file.path.stem}{from_file.format.extensions[0]}"
        return data

    def training(self, data:DstSeqData):
        data = self.preprocess(data)
        sequences = data.state_sequences()
        print(f"Training on {len(sequences)} sequences!")
        for i, results in enumerate(self.seq2seq_model.training(sequences)):
            self.iteration += 1
            self.record_hyperparameters('training', data=repr(data))
            results = DstSeqResults(**vars(results))
            if data.file is not None:
                results.file = self.path_prefix() /\
                               f"{data.file.path.stem}.train_results.pkl"
            yield results

    def train(self, data:DstSeqData):
        return list(self.training(data))



if __name__ == '__main__':

    # data = DstSeqData(
    #     [
    #         ('I want to buy a red car', dict(item='car', color='red', time=None)),
    #         ('I want to buy a blue car', dict(item='car', color='blue', time=None)),
    #         ('Put tom down for 6 oclock', dict(time='6', attendee='tom', item=None)),
    #     ],
    #     file='toy.txt'
    # )



    def train(
        train_data:str,
        valid_data:str,
        directory='ex',
        experiment:str|None=None,
        checkpoint='t5-small',
        load_in_8bit=False,
        epochs=5,
        learning_rate=1e-4,
        weight_decay=0.0,
        train_batch_size=2,
        gradient_accumulation_steps=64,
        predict_batch_size=2,
        max_length=2048,
        max_new_tokens=256,
        gen_beams=1,
        gen_sampling=False,
        repetition_penalty:float=1.1,
        repetition_alpha:float=None,
        max_context_turns:int=None,
        sep_token='|',
        eoi_token='->',
        req_token='?',
        device='cuda',
        train_dialogues_limit:int=None,
        valid_dialogues_limit:int=None,
        silence_training=False,
        saving: bool = True,
        email_notifications: str = 'jamesfinch293@gmail.com',
        catch_and_alert_errors: bool = True
    ):
        from dst.approaches.t5 import T5
        train_data = DstSeqData.load(train_data)
        valid_data = DstSeqData.load(valid_data)
        if train_dialogues_limit is not None:
            train_data = DstSeqData(
                train_data.dialogues[:train_dialogues_limit],
                ontology=train_data.ontology,
                file=train_data.file
            )
        if valid_dialogues_limit is not None:
            valid_data = DstSeqData(
                valid_data.dialogues[:valid_dialogues_limit],
                ontology=valid_data.ontology,
                file=valid_data.file
            )
        dst = Seq2seqDSG(T5(
            checkpoint=checkpoint,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            train_batch_size=train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            predict_batch_size=predict_batch_size,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            gen_beams=gen_beams,
            gen_sampling=gen_sampling,
            repetition_penalty=repetition_penalty,
            repetition_alpha=repetition_alpha,
            device=device,
            silence_training=silence_training,
            load_in_8bit=load_in_8bit,
        ),
            directory=directory,
            experiment=experiment,
            max_context_turns=max_context_turns,
            sep_token=sep_token,
            eoi_token=eoi_token,
            req_token=req_token,
        )
        if saving:
            dst.hyperparameters.save()
        training = dst.training(train_data)
        if email_notifications:
            ez.email(
                email_notifications,
                f'DSG {dst.experiment} started',
                f"{dst.hyperparameters.display()}"
            )
        try:
            while training:
                with ez.Timer() as epoch_time:
                    with ez.Timer() as train_time:
                        try:
                            train_results = next(training)
                        except StopIteration:
                            break
                    train_logits = dst.seq2seq_model.logits(train_data.state_sequences())
                    train_results.update(train_logits)
                    print(f"Epoch {train_results.epoch} in {train_time.display}")
                    with ez.Timer() as predict_time:
                        predictions = dst.predict(valid_data)
                    if saving:
                        predictions.save()
                    print(f"Eval {train_results.epoch} (predictions in {predict_time.display})")
                    valid_results = DstSeqResults(
                        _file=f"{train_results.file.path.parent}"
                              f"/{train_results.file.path.stem.split('.')[0]}"
                              f".valid_results.pkl"
                    )
                    valid_results.update(predictions)
                    with ez.Timer() as logits_time:
                        valid_logits = dst.seq2seq_model.logits(valid_data.state_sequences())
                    valid_results.update(valid_logits)
                    print(f"    Train Loss: {train_results.loss:.4f}")
                    print(f"    Slot Update Accuracy: {valid_results.slot_update_accuracy:.4f}")
                    print(f"    Joint Goal Accuracy: {valid_results.joint_goal_accuracy:.4f}")
                    print(f"    Perplexity: {valid_results.perplexity:.4f} (logits in {logits_time.display})")
                    print(f"    Train Perplexity: {train_results.perplexity:.4f}")
                    # for turn in [turn for dialogue in valid_data.dialogues for turn in dialogue.turns][:3]:
                    #     print(turn.display())
                    #     print()
                    #     print(turn.state_sequence.seq_input)
                    #     print(turn.state_sequence.seq_label)
                    #     print(turn.state_sequence.seq_output)
                    #     print('\n\n')
                    if saving:
                        train_results.save()
                        valid_results.save()
                        dst.save()
                        dst.hyperparameters.save()
                if email_notifications:
                    ez.email(
                        email_notifications,
                        f'DSG {dst.experiment} on {train_data.file.path.name[:4]} did {dst.iteration} epochs',
                        '\n'.join(
                            [
                                f'Time: {epoch_time.display}\n\n',
                                'Valid',
                                '=====',
                                '\n'.join(
                                    f"{metric}: {str(value)[:30] + ('...' if len(str(value)) > 30 else '')}"
                                        for metric, value in vars(valid_results).items()
                                ),
                                '\n\n'
                                'Train',
                                '=====',
                                '\n'.join(
                                    f"{metric}: {value}" for metric, value in vars(train_results).items()
                                ),
                                '\n\n',
                                f"{dst.hyperparameters.display()}"
                            ]
                        )
                    )
        except (Exception if catch_and_alert_errors else NoException) as e:
            if email_notifications:
                ez.email(
                    email_notifications,
                    f'DSG {dst.experiment} error',
                    traceback.format_exc()
                )
            traceback.print_exc()
            print('\n\n', f"{dst.hyperparameters.display()}")

    fire.Fire(train)


    # train(
    #     # train_data='data/mwz2.4/mwoz24_train_train.pkl',
    #     # valid_data='data/mwz2.4/mwoz24_test_train.pkl',
    #     train_data='data/gptdst5k/gptdst5k_train_domains_0.pkl',
    #     valid_data='data/gptdst5k/gptdst5k_train_domains_0.pkl',
    #     directory='ex',
    #     experiment=None,
    #     checkpoint='t5-small',
    #     load_in_8bit=False,
    #     epochs=300,
    #     learning_rate=1e-3,
    #     weight_decay=0.0,
    #     train_batch_size=2,
    #     gradient_accumulation_steps=64,
    #     predict_batch_size=2,
    #     max_length=2048,
    #     max_new_tokens=256,
    #     gen_beams=1,
    #     gen_sampling=False,
    #     repetition_penalty=1.1,
    #     repetition_alpha=None,
    #     sep_token='|',
    #     eoi_token='->',
    #     req_token='?',
    #     device='cuda',
    #     train_dialogues_limit=1,
    #     valid_dialogues_limit=1,
    #     silence_training=False,
    #     email_notifications=False,
    #     saving=False
    # )
