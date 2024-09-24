import itertools

import ezpyzy as ez
import dataclasses as dc
import dsi.preprocessing as dpp
import random as rng
import typing as T
import copy as cp


@dc.dataclass
class Turn(ez.Table):
    text: ez.ColStr = None
    dialogue: ez.ColStr = None
    turn_index: ez.ColInt = None
    speaker: ez.ColStr = None
    domain: ez.ColStr = None
    turn_id: ez.ColID = None

    def continuation(self, turn_id):
        dialogue_index_to_turn_id = {(d, i): t for d, i, t in zip(
            self.dialogue, self.turn_index, self.turn_id
        )}
        turn_id_to_dialogue_index = {t: (d, i) for d, i, t in zip(
            self.dialogue, self.turn_index, self.turn_id
        )}
        dialogue, index = turn_id_to_dialogue_index[turn_id]
        continuation = [turn_id]
        while (dialogue, index+1) in dialogue_index_to_turn_id:
            index += 1
            continuation.append(dialogue_index_to_turn_id[(dialogue, index)])
        return self[continuation]

@dc.dataclass
class SlotValue(ez.Table):
    slot: ez.ColStr = None
    value: ez.ColStr = None
    turn_id: ez.ColStr = None
    slot_id: ez.ColStr = None
    slot_value_id: ez.ColID = None

@dc.dataclass
class Slot(ez.Table):
    slot: ez.ColStr = None
    domain: ez.ColStr = None
    description: ez.ColStr = None
    slot_id: ez.ColID = None

@dc.dataclass
class ValueCandidate(ez.Table):
    candidate_value: ez.ColStr = None
    slot_id: ez.ColStr = None
    is_provided: ez.ColBool = None
    value_candidate_id: ez.ColID = None

@dc.dataclass
class QuestionAnswerPair(ez.Table):
    question: ez.ColStr = None
    answer: ez.ColStr = None
    slot_value_id: ez.ColStr = None

@dc.dataclass
class Prediction(ez.Table):
    slot_value_id: ez.ColID = None
    prompt: ez.ColStr = None
    generated: ez.ColStr = None
    prediction: ez.ColStr = None
    slot_id: ez.ColStr = None


class DstExample(T.NamedTuple):
    text: str
    speaker: str
    domain: str
    predicted_state: dict[str, str]
    actual_state: dict[str, str]
    prompts: list[str|None]
    generations: list[str|None]

class DstIdExample(T.NamedTuple):
    text: str
    speaker: str
    domain: str
    predicted_state: dict[str, str]
    actual_state: dict[str, str]
    prompts: list[str|None]
    generations: list[str|None]
    turn_id: str

class DstSample(T.NamedTuple):
    text: str
    speaker: str
    domain: str
    dialogue: str
    turn_id: str
    turn_index: int
    history: list[str]
    context: list[str]
    state: dict[str, str]

@dc.dataclass
class DstState:
    turn_id: str
    dialogue_id: str
    index: int
    text: str
    speaker: str
    domain: str
    predicted_state: dict[str, str]
    actual_state: dict[str, str]
    accumulation: dict[str, list[tuple[str, str]]] # slot -> list of value, svid
    prompts: dict[str, str]
    generations: dict[str, str]


@dc.dataclass
class Data:
    data_path: str = None
    prediction_path: str = None
    slots: Slot = None
    value_candidates: ValueCandidate = None

    def __post_init__(self):
        if self.data_path:
            self.turns = Turn.of(f"{self.data_path}/turn.csv")
            self.slot_values = SlotValue.of(f"{self.data_path}/slot_value.csv")
            if self.slots is None:
                self.slots = Slot.of(f"{self.data_path}/slot.csv")
            if self.value_candidates is None:
                self.value_candidates = ValueCandidate.of(f"{self.data_path}/value_candidate.csv")
            if self.prediction_path is not None:
                self.predictions = Prediction.of(self.prediction_path)
            else:
                self.predictions = None
        else:
            self.turns = Turn.of([], fill=None)
            self.slot_values = SlotValue.of([], fill=None)
            self.slots = Slot.of([], fill=None)
            self.value_candidates = ValueCandidate.of([], fill=None)
            self.predictions = Prediction.of([], fill=None)

    def copy(self):
        new_data = Data()
        new_data.data_path = self.data_path
        new_data.prediction_path = self.prediction_path
        new_data.turns = Turn.of(self.turns)
        new_data.slot_values = SlotValue.of(self.slot_values)
        new_data.slots = Slot.of(self.slots)
        new_data.value_candidates = ValueCandidate.of(self.value_candidates)
        new_data.predictions = Prediction.of(self.predictions) if self.predictions else None
        return new_data

    def save(self, folder):
        self.turns().save(f"{folder}/turn.csv")
        self.slot_values().save(f"{folder}/slot_value.csv")
        self.slots().save(f"{folder}/slot.csv")
        self.value_candidates().save(f"{folder}/value_candidate.csv")
        if self.predictions:
            self.predictions().save(f"{folder}/predictions.csv")

    @classmethod
    def of(cls, dialogues:str|list[str]|list[list[str]], slots:dict):
        slots = Slot.of(dict(slot=list(slots.keys()), description=list(slots.values())), fill=None)
        slots.slot_id = ez.IDColumn(slots.slot_id)
        if isinstance(dialogues, str):
            dialogues = [[dialogues]]
        elif any(isinstance(d, str) for d in dialogues):
            dialogues = [dialogues]
        turns = []
        for dialogue, dialogue_id in zip(dialogues, ez.digital_iteration()):
            for i, turn in enumerate(dialogue):
                turns.append(dict(
                    text=turn,
                    dialogue=dialogue_id,
                    turn_index=i,
                ))
        turns = Turn.of(turns, fill=None)
        data = Data()
        data.turns = turns
        data.slots = slots
        dpp.add_neg_slot_targets(data)
        return data

    def examples(self, n=None, accumulate_states=True, with_turn_ids=False) -> T.Iterable[T.Iterable[DstExample]]:
        self.turns().sort(self.turns.turn_index)
        examples = self.predictions.slot_value_id << self.slot_values.slot_value_id
        turns_with_predictions = set(examples.turn_id)
        dialogues_with_predictions = set(
            self.turns.dialogue[[
                x in turns_with_predictions for x in self.turns.turn_id
        ]])
        if n is not None:
            dialogues_with_predictions = set(rng.sample(dialogues_with_predictions, n))
        turn_id_to_everything = {
            tid: (dialogue, speaker, text, turn_index, domain)
            for tid, dialogue, text, turn_index, speaker, domain in zip(
                self.turns.turn_id, self.turns.dialogue, self.turns.text, self.turns.turn_index, self.turns.speaker, self.turns.domain
        )}
        dialogue_to_turn_ids = {dialogue: [] for dialogue in self.turns.dialogue}
        for dialogue, tid in zip(self.turns.dialogue, self.turns.turn_id):
            dialogue_to_turn_ids[dialogue].append(tid)
        turns_to_slot_value_ids = {tid: [] for tid in self.turns.turn_id}
        for turn_id, slot_value_id in zip(examples.turn_id, examples.slot_value_id):
            turns_to_slot_value_ids[turn_id].append(slot_value_id)
        svid_to_slot_value = {svid: (slot, value) for svid, slot, value in zip(
            examples.slot_value_id, examples.slot, examples.value
        )}
        svid_to_prompt_generation = {svid: (prompt, gen) for svid, prompt, gen in zip(
            examples.slot_value_id, examples.prompt, examples.generated
        )}
        predictions_for_slot_values = {
        svid: (pred, gen) for svid, pred, gen in zip(
            self.predictions.slot_value_id, self.predictions.prediction, self.predictions.generated
        )}
        for dialogue, tids in dialogue_to_turn_ids.items():
            if dialogue in dialogues_with_predictions:
                actual_state = {}
                predicted_state = {}
                def turn_iterator():
                    for tid in tids:
                        _, speaker, text, turn_index, domain = turn_id_to_everything[tid]
                        svids = turns_to_slot_value_ids[tid]
                        actual_state_update = {}
                        predicted_state_update = {}
                        prompts = []
                        generations = []
                        for svid in svids:
                            if svid in predictions_for_slot_values:
                                slot, value = svid_to_slot_value[svid]
                                prediction, _ = predictions_for_slot_values[svid]
                                if value is not None:
                                    actual_state_update[slot] = value
                                if prediction is not None:
                                    predicted_state_update[slot] = prediction
                            prompt, generation = svid_to_prompt_generation.get(svid, (None, None))
                            prompts.append(prompt)
                            generations.append(generation)
                        actual_state.update(actual_state_update)
                        predicted_state.update(predicted_state_update)
                        if not accumulate_states:
                            if not with_turn_ids:
                                yield DstExample(
                                    text, speaker, domain, predicted_state_update, actual_state_update,
                                    prompts, generations
                                )
                            else:
                                yield DstIdExample(
                                    text, speaker, domain, predicted_state_update, actual_state_update,
                                    prompts, generations, tid
                                )
                        else:
                            if not with_turn_ids:
                                yield DstExample(
                                    text, speaker, domain, dict(predicted_state), dict(actual_state),
                                    prompts, generations
                                )
                            else:
                                yield DstIdExample(
                                    text, speaker, domain, dict(predicted_state), dict(actual_state),
                                    prompts, generations, tid
                                )
                yield turn_iterator()


    def samples(self, accumulate_state=False) -> T.Iterable[T.Iterable[DstSample]]:
        self.turns().sort(self.turns.turn_index)
        examples = self.slot_values
        turn_id_to_everything = {
            tid: (dialogue, speaker, text, turn_index, domain)
            for tid, dialogue, text, turn_index, speaker, domain in zip(
                self.turns.turn_id, self.turns.dialogue, self.turns.text, self.turns.turn_index, self.turns.speaker, self.turns.domain
        )}
        dialogue_to_turn_ids = {dialogue: [] for dialogue in self.turns.dialogue}
        for dialogue, tid in zip(self.turns.dialogue, self.turns.turn_id):
            dialogue_to_turn_ids[dialogue].append(tid)
        turns_to_slot_value_ids = {tid: [] for tid in self.turns.turn_id}
        for turn_id, slot_value_id in zip(examples.turn_id, examples.slot_value_id):
            turns_to_slot_value_ids[turn_id].append(slot_value_id)
        svid_to_slot_value = {svid: (slot, value) for svid, slot, value in zip(
            examples.slot_value_id, examples.slot, examples.value
        )}
        for dialogue, tids in dialogue_to_turn_ids.items():
            actual_state = {}
            def turn_iterator():
                context = []
                for i, tid in enumerate(tids):
                    _, speaker, text, turn_index, domain = turn_id_to_everything[tid]
                    context.append(text)
                    svids = turns_to_slot_value_ids[tid]
                    actual_state_update = {}
                    for svid in svids:
                        slot, value = svid_to_slot_value[svid]
                        if value is not None:
                            actual_state_update[slot] = value
                    actual_state.update(actual_state_update)
                    yield DstSample(
                        text, speaker, domain, dialogue, tid, i, tids[:i], list(context),
                        dict(actual_state) if accumulate_state else actual_state_update
                    )
            yield turn_iterator()


    def states(self, n=None):
        if self.predictions:
            predictions = {svid: (p, g, v) for svid, p, g, v in zip(
                self.predictions.slot_value_id,
                self.predictions.prompt,
                self.predictions.generated,
                self.predictions.prediction
            )}
        else:
            predictions = {}
        exs = self.slot_values.turn_id << self.turns.turn_id
        exs().sort([
            (d, i, s) for d, i, s in zip(exs.dialogue, exs.turn_index, exs.slot)
        ])
        turns = []
        dialogues = [turns]
        for tid, did, dom, tidx, spkr, text, svid, s, v in list(zip(
            exs.turn_id, exs.dialogue, exs.domain, exs.turn_index, exs.text, exs.speaker,
            exs.slot_value_id, exs.slot, exs.value
        )):
            if not turns:
                turns.append(DstState(tid, did, tidx, text, spkr, dom, {}, {}, {}, {}, {}))
            elif dialogues[-1][-1].dialogue_id != did:
                turns = [DstState(tid, did, tidx, text, spkr, dom, {}, {}, {}, {}, {})]
                dialogues.append(turns)
            elif turns[-1].turn_id != tid:
                previous = turns[-1]
                turns.append(DstState(tid, did, tidx, text, spkr, dom,
                    dict(previous.predicted_state), dict(previous.actual_state),
                    cp.deepcopy(previous.accumulation),
                    dict(previous.prompts), dict(previous.generations)
                ))
            turn = turns[-1]
            turn.accumulation.setdefault(s, []).append((v, svid))
            if v is not None:
                turn.actual_state[s] = v
            if svid in predictions:
                prompt, generated, predicted = predictions[svid]
                turn.prompts[svid] = prompt
                turn.generations[svid] = prompt
                turn.predicted_state[s] = predicted
        return rng.sample(dialogues, n) if n else dialogues




if __name__ == '__main__':
    with ez.check('Loading dsg5k...'):
        data = Data('data/dsg5k/train')
    print(data.turns[:5]().display(30), '\n')
    print(data.slot_values[:5]().display(30), '\n')
    print(data.slots[:5]().display(30))




