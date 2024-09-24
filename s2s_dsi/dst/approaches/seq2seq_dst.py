import copy
import functools
import pathlib
import random
import itertools
import traceback
import fire
import ezpyz as ez
from dst.approaches.dst_seq_data import DstSeqData, DstSeqResults, Turn
from dst.data.dst_data import Slot
from dst.approaches.sequences_data import SequencesData, SequenceResults
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


def old_prompt(turn:Turn, slot:Slot):
    context = turn.context()
    if slot.description:
        description = f" - {slot.description}"
    else:
        description = ""
    seq_in = ''
    for speaker, context_turn in reversed(
        list(zip(itertools.cycle('AB'), context[-3:]))
    ):
        seq_in = f"@ {speaker}: {context_turn.turn}\n" + seq_in
    seq_in += f"* {slot.name}{description} ->"
    return seq_in


class Seq2seqDst:
    remove_label_from_open_rate: float = 0.0
    def __init__(self,
        seq2seq_model:Seq2SeqModel,
        directory=None,
        approach=None,
        experiment=None,
        prompt=None,
        context_window_limit=None,
        include_slot_description=True,
        example_slot_values_limit=None,
        example_slot_values_variance=0,
        categorical_as_open_rate=None,
        open_as_categorical_rate=None,
        remove_label_from_open_rate=None,
        pre_shuffle_example_candidates=False,
        post_shuffle_examples=False,
        sep_token='**',
        etc_token='etc',
        match_ontology_candidates=False,
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
        self.context_window_limit = context_window_limit
        self.include_slot_description = include_slot_description
        self.example_slot_values_limit = example_slot_values_limit
        self.example_slot_values_variance = example_slot_values_variance
        self.categorical_as_open_rate = categorical_as_open_rate
        self.open_as_categorical_rate = open_as_categorical_rate
        self.pre_shuffle_example_candidates = pre_shuffle_example_candidates
        self.post_shuffle_examples = post_shuffle_examples
        self.remove_label_from_open_rate = remove_label_from_open_rate
        self.sep_token = sep_token
        self.etc_token = etc_token
        self.prompt:callable = self.create_prompt if prompt is None else prompt
        self.iteration = 0
        self.hyperparameters = Hyperparameters(
            pathlib.Path(
                self.directory, self.approach, self.experiment, str(self.iteration),
                f"hyperparameters.pkl"
            )
        )
        self.record_hyperparameters('init')

    def save(self, path:ez.filelike=None):
        if path is None:
            path = pathlib.Path(self.path_prefix(), 'model')
        path = ez.File(path).path
        self.seq2seq_model.save(path)
        scopy = copy.copy(self)
        scopy.seq2seq_model = None
        scopy.prompt = 'old' if scopy.prompt is old_prompt else None
        ez.Cache(path/'Seq2seqDst.pkl').save(scopy)
        self.hyperparameters.save(path/'hyperparameters.pkl')

    @classmethod
    def load(cls, path:ez.filelike, directory=None, device=None) -> 'Seq2seqDst':
        path = ez.File(path).path
        seq2seq_dst = ez.Cache(path/'Seq2seqDst.pkl').load()
        seq2seq_dst.prompt = {
            None:seq2seq_dst.create_prompt
        }.get(getattr(seq2seq_dst, 'prompt', None), old_prompt)
        seq2seq = ez.Cache(path/'Seq2seq.pkl').load()
        seq2seq = seq2seq.load(path=path, directory=directory, device=device)
        seq2seq_dst.seq2seq_model = seq2seq
        return seq2seq_dst

    def record_hyperparameters(self, stage, **kwargs):
        self.hyperparameters.record(self, stage=stage, **kwargs)
        self.hyperparameters[-1] = {
            **self.seq2seq_model.hyperparameters[-1], **self.hyperparameters[-1]
        }

    def path_prefix(self):
        return pathlib.Path(self.directory) / self.approach / self.experiment / str(self.iteration)

    def create_prompt(self, turn: Turn = None, slot: Slot = None):
        context = turn.context()
        if isinstance(self.context_window_limit, int):
            context = context[-self.context_window_limit:]
        seq_in = ''
        for speaker, context_turn in reversed(
            list(zip(itertools.cycle(['A', 'B']), context))
        ):
            seq_in = f"{speaker}: {context_turn.turn}\n" + seq_in
        categorical = slot.categorical
        if categorical and self.categorical_as_open_rate is not None:
            if random.random() < self.categorical_as_open_rate:
                categorical = False
        elif not categorical and self.open_as_categorical_rate is not None:
            if random.random() < self.open_as_categorical_rate:
                categorical = True
        examples = [
            candidate for candidate in slot.values or []
            if candidate not in {'etc', '...', 'etc.', 'Etc', 'Etc.'}
        ]
        if categorical:
            slot_values = {slot.name: values for slot, values in turn.slots.items()} if turn.slots else {}
            label_values = set(slot_values.get(slot.name, []) or [])
            candidate_set = [
                candidate for candidate in examples
                if candidate not in label_values
            ]
            examples = list(label_values) + candidate_set
        elif self.remove_label_from_open_rate is not None:
            if random.random() < self.remove_label_from_open_rate:
                slot_values = {slot.name: values for slot, values in turn.slots.items()} if turn.slots else {}
                label_values = set(slot_values.get(slot.name, []) or [])
                examples = [
                    candidate for candidate in examples
                    if candidate not in label_values
                ]
        if self.pre_shuffle_example_candidates:
            random.shuffle(examples)
        example_slot_values_limit = self.example_slot_values_limit
        if self.example_slot_values_limit is not None:
            if self.example_slot_values_variance is not None:
                example_slot_values_limit = random.randint(
                    self.example_slot_values_limit - self.example_slot_values_variance,
                    self.example_slot_values_limit + self.example_slot_values_variance
                )
            examples = examples[:example_slot_values_limit]
        if self.post_shuffle_examples:
            random.shuffle(examples)
        if not categorical:
            example_tail = f', {self.etc_token}'
        else:
            example_tail = ''
        examples = f" ({', '.join(examples) + example_tail})" if examples else ''
        description = f": {slot.description}" if self.include_slot_description else ''
        seq_in += f"\n{self.sep_token} {slot.name}{description}{examples}"
        return seq_in

    def preprocess(self, data:DstSeqData) -> DstSeqData:
        timer = ez.Timer()
        for dialogue in data.dialogues:
            for turn in [turn for turn in dialogue.turns if turn.slots is not None]:
                for slot, seqs in turn.slot_sequences.items():
                    seq_in = self.prompt(turn, slot) # noqa
                    seqs.seq_input = seq_in
                    seqs.seq_label = ', '.join(turn.slots[slot]) if turn.slots[slot] else ''
        if data.file is not None:
            data.file = self.path_prefix().parent / 'preprocess' /\
                        f"{data.file.path.stem}{data.file.format.extensions[0]}"
        timer.stop()
        return data

    def postprocess(self, data:DstSeqData):
        for dialogue in data.dialogues:
            for turn in (turn for turn in dialogue.turns if turn.slots is not None):
                turn.predicted_slots = {}
                for slot, seqs in turn.slot_sequences.items():
                    if not seqs.seq_output:
                        predicted = None
                    else:
                        predicted = seqs.seq_output.split(', ')
                    turn.predicted_slots[slot] = predicted
        if data.file is not None:
            data.file = self.path_prefix().parent / 'postprocess' /\
                        f"{data.file.path.stem}{data.file.format.extensions[0]}"
        return data

    def predict(self, data:DstSeqData):
        from_file = data.file
        data = self.preprocess(data)
        sequences = data.slot_sequences()
        predicted_sequences = self.seq2seq_model.predict(sequences)
        data = self.postprocess(data)
        if from_file is not None:
            data.file = self.path_prefix() / 'predictions' /\
                        f"{from_file.path.stem}{from_file.format.extensions[0]}"
        return data

    def training(self, data:DstSeqData):
        data = self.preprocess(data)
        sequences = data.slot_sequences()
        for i, results in enumerate(self.seq2seq_model.training(sequences)):
            self.iteration += 1
            self.record_hyperparameters('training', data=repr(data))
            results = DstSeqResults(**vars(results))
            if data.file is not None:
                results.file = self.path_prefix() /\
                               f"{data.file.path.stem}.train_results.pkl"
            yield results
            data = self.preprocess(data)
            for new, old in zip(data.slot_sequences(), sequences):
                old.seq_input = new.seq_input

    def train(self, data:DstSeqData):
        return list(self.training(data))


def train(
    # train_data='data/mwz2.4/mwoz24_train_attraction.pkl',
    # valid_data='data/mwz2.4/mwoz24_test_attraction.pkl',
    # train_data='data/mwz2.4/mwoz24_train_hotel.pkl',
    # valid_data='data/mwz2.4/mwoz24_test_hotel.pkl',
    # train_data='data/mwz2.4/mwoz24_train_restaurant.pkl',
    # valid_data='data/mwz2.4/mwoz24_test_restaurant.pkl',
    # train_data='data/mwz2.4/mwoz24_train_taxi.pkl',
    # valid_data='data/mwz2.4/mwoz24_test_taxi.pkl',
    train_data='data/mwz2.4/mwoz24_train_train.pkl',
    # valid_data='data/mwz2.4/mwoz24_test_train.pkl',
    #
    valid_data='data/mwz2.4/mwoz24_train_train.pkl',

    # train_data='data/gptdst5k/gptdst5k_train_domains_0.pkl',
    # valid_data='data/gptdst5k/gptdst5k_valid_domains_0.pkl',

    directory:str='ex',
    experiment:str=None,

    # checkpoint='ex/Seq2seqDst/WiseCount/1/model',
    checkpoint:str='t5-small',

    epochs:int=5,
    learning_rate:float=1e-4,
    weight_decay:float=0.0,
    train_batch_size:int=8,
    gradient_accumulation_steps:int=16,
    predict_batch_size:int=8,
    max_length:int=2048,
    max_new_tokens:int=32,
    gen_beams:int=1,
    gen_sampling:bool=False,
    repetition_penalty:float=None,
    repetition_alpha:float=None,
    prompt=None,
    context_window_limit:int=None,
    include_slot_description:bool=True,
    example_slot_values_limit:int=None,
    example_slot_values_variance:int=0,
    categorical_as_open_rate:float=None,
    open_as_categorical_rate:float=None,
    remove_label_from_open_rate:float=None,
    pre_shuffle_example_candidates:bool=False,
    post_shuffle_examples:bool=False,
    sep_token:str='**',
    etc_token:str='etc',
    device:str='cuda',
    train_dialogues_limit:int=None,
    valid_dialogues_limit:int=None,
    silence_training:bool=False,
    saving:bool=True,
    email_notifications:bool=True,
    catch_and_alert_errors:bool=True
):
    from dst.approaches.t5 import T5
    train_data = DstSeqData.load(train_data)
    valid_data = DstSeqData.load(valid_data)
    if train_dialogues_limit is not None:
        train_data = DstSeqData(
            train_data.dialogues[:int(train_dialogues_limit)],
            ontology=train_data.ontology,
            file=train_data.file
        )
    if valid_dialogues_limit is not None:
        valid_data = DstSeqData(
            valid_data.dialogues[:int(valid_dialogues_limit)],
            ontology=valid_data.ontology,
            file=valid_data.file
        )
    dst = Seq2seqDst(
        T5(
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
        ),
        directory=directory,
        experiment=experiment,
        prompt=dict(old_prompt=old_prompt).get(prompt),
        context_window_limit=context_window_limit,
        include_slot_description=include_slot_description,
        example_slot_values_limit=example_slot_values_limit,
        example_slot_values_variance=example_slot_values_variance,
        categorical_as_open_rate=categorical_as_open_rate,
        open_as_categorical_rate=open_as_categorical_rate,
        remove_label_from_open_rate=remove_label_from_open_rate,
        pre_shuffle_example_candidates=pre_shuffle_example_candidates,
        post_shuffle_examples=post_shuffle_examples,
        sep_token=sep_token,
        etc_token=etc_token
    )
    if saving:
        dst.hyperparameters.save()
    training = dst.training(train_data)
    if email_notifications:
        ez.email(
            'jamesfinch293@gmail.com',
            f'DST {dst.experiment} started',
            f"{dst.hyperparameters.display()}"
        )
    try:
        while training:
            with ez.Timer() as epoch_time:
                with ez.Timer() as train_time:
                    try: train_results = next(training)
                    except StopIteration: break
                train_logits = dst.seq2seq_model.logits(train_data.slot_sequences())
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
                    valid_logits = dst.seq2seq_model.logits(valid_data.slot_sequences())
                valid_results.update(valid_logits)
                print(f"    Train Loss: {train_results.loss:.4f}")
                print(f"    Slot Accuracy: {valid_results.slot_accuracy:.4f}")
                print(f"    Joint Goal Accuracy: {valid_results.joint_goal_accuracy:.4f}")
                print(f"    Perplexity: {valid_results.perplexity:.4f} (logits in {logits_time.display})")
                print(f"    Train Perplexity: {train_results.perplexity:.4f}")
                if saving:
                    train_results.save()
                    valid_results.save()
                    dst.save()
            if email_notifications:
                ez.email(
                    'jamesfinch293@gmail.com',
                    f'DST {dst.experiment} on {train_data.file.path.name[:4]} did {dst.iteration} epochs',
                    '\n'.join([
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
                    ])
                )
    except (Exception if catch_and_alert_errors else NoException) as e:
        if email_notifications:
            ez.email(
                'jamesfinch293@gmail.com',
                f'DST {dst.experiment} error',
                traceback.format_exc() + '\n\n' + dst.hyperparameters.display()
            )
        traceback.print_exc()


if __name__ == '__main__':

    # data = DstSeqData(
    #     [
    #         ('I want to buy a red car', dict(item='car', color='red', time=None)),
    #         ('I want to buy a blue car', dict(item='car', color='blue', time=None)),
    #         ('Put tom down for 6 oclock', dict(time='6', attendee='tom', item=None)),
    #     ],
    #     file='toy.txt'
    # )

    fire.Fire(train)

    # train(
    #     train_data='data/gptdst5k/gptdst5k_train_domains_0.pkl',
    #     valid_data='data/gptdst5k/gptdst5k_valid_domains_0.pkl',
    #     learning_rate=1e-4,
    #     train_batch_size=2,
    #     gradient_accumulation_steps=2,
    #     predict_batch_size=2,
    #     example_slot_values_limit=16,
    #     example_slot_values_variance=4,
    #     open_as_categorical_rate=0.2,
    #     remove_label_from_open_rate=0.8,
    #     post_shuffle_examples=True,
    #     email_notifications=False,
    #     saving=False,
    #     train_dialogues_limit=1,
    #     valid_dialogues_limit=1,
    #     epochs=2,
    # )