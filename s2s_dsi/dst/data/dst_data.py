
from __future__ import annotations
from dst.data.results import Results
import random
import math
import typing as T
import dataclasses
import ezpyz as ez
import itertools
import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)

from bert_score import score as bertscore


slotlike: T.TypeAlias = T.Union[
    str,                        # name
    tuple[str, str],            # name, description
    tuple[str, list[str]],      # name, values
    tuple[str, str, list[str]], # name, description, values
    'Slot'
]
@dataclasses.dataclass(eq=False)
class Slot:
    categorical:bool=None
    def __init__(self,
        name: slotlike,
        description: str | None = None,
        domain: str|None = None,
        values: list[str]|None = None,
        categorical:bool|None = None
    ):
        if isinstance(name, Slot):
            self.name = name.name # noqa
            self.description = name.description # noqa
            self.domain = name.domain # noqa
            self.values = name.values # noqa
            self.categorical = name.categorical # noqa
            return
        if isinstance(name, tuple) and name:
            for field in name[1:]:
                if isinstance(field, str):
                    description = field
                elif isinstance(field, list):
                    values = field
            name = name[0]
        self.name:str = name
        self.description:str|None = description
        self.domain:str|None = domain
        self.values:list[str]|None = values
        self.categorical:bool = categorical
        if self.categorical is None and self.values:
            self.categorical = not (set(self.values) & {'Etc', 'etc', 'etc.', 'Etc.', '...'})

    def key(self):
        return self.name, self.description, self.domain

    def __eq__(self, other):
        return self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __str__(self):
        return f'{type(self).__name__}({self.name})'
    __repr__ = __str__

ontologylike: T.TypeAlias = T.Union[
    list[slotlike],
    dict[str,str|list[str]|tuple[str,list[str]]],
    'Ontology'
]
class Ontology:

    def __init__(
        self,
        ontology:ontologylike=None,
        **domains:ontologylike
    ):
        self.slot_ids: dict[tuple[str, str | None, str | None], Slot] = getattr(
            self, 'slot_ids', {}
        )
        if ontology is not None:
            domains[None] = ontology  # noqa
        for domain, slots in domains.items():
            if isinstance(slots, dict):
                slotlist = []
                for name, definition in slots.items():
                    if isinstance(definition, str):
                        slotlist.append((name, definition, None))
                    elif isinstance(definition, list):
                        slotlist.append((name, None, definition))
                    elif isinstance(definition, tuple):
                        slotlist.append((name, *definition))
                for slot in slotlist:
                    self.add(slot, domain=domain)
            elif isinstance(slots, Ontology):
                self.slot_ids = slots.slot_ids
                for slot in ontology.slots():
                    self.add(slot)
    update = __init__

    def add(self, slot: slotlike, domain=None):
        slot = Slot(slot, domain=domain if domain is not None else getattr(slot, 'domain', None))
        slot_id = (slot.name, slot.description, slot.domain)
        if slot_id in self.slot_ids:
            vars(self.slot_ids[slot_id]).update({k:v for k, v in vars(slot).items() if v is not None})
        else:
            self.slot_ids[slot_id] = slot
        return self.slot_ids[slot_id]

    def slots(self) -> list[Slot]:
        return list(self.slot_ids.values())

    def domains(self) -> dict[str, list[Slot]]:
        domains = {}
        for slot in self.slots():
            domains.setdefault(slot.domain, []).append(slot)
        return domains

    def __str__(self):
        return f'<{type(self).__name__} object with {len(self.slot_ids)} slots>'
    __repr__ = __str__


turnlike: T.TypeAlias = T.Union[
    str,                                        # text
    tuple[str, dict[slotlike, str|list[str]]],  # text, slots
    tuple[str, dict[slotlike, str|list[str]], dict[slotlike, str|list[str]]], # text, slots, predicted slots
    'Turn'
]
class Turn:
    def __init__(self,
        turn:turnlike,
        speaker=None,
        listener=None,
        slots:dict[slotlike, str|list[str]]=None,
        predicted_slots:dict[slotlike, str|list[str]]=None,
        dialogue=None,
        index=None
    ):
        self.turn:str = None # noqa
        self.speaker:str|None = speaker
        self.listener:str|None = listener
        self.slots:dict[Slot, list[str]]|None = None
        self.predicted_slots:dict[Slot, list[str]]|None = predicted_slots
        if slots is not None:
            self.slots = {}
            for slot, value in slots.items():
                if not isinstance(slot, Slot):
                    slot = Slot(slot)
                if isinstance(value, str):
                    value = [value]
                self.slots[slot] = value
        self.dialogue:T.Optional['Dialogue'] = dialogue
        self.index:int|None = index
        if isinstance(turn, Turn):
            vars(self).update({
                k:v for k, v in vars(turn).items()
                if v is not None and getattr(self, k, None) is None
            })
        elif isinstance(turn, str):
            self.turn = turn
        elif isinstance(turn, tuple):
            self.turn = turn[0]
            self.slots:dict[Slot, list[str]]|None = {}
            for slot, value in turn[1].items():
                slot = Slot(slot)
                if isinstance(value, str):
                    value = [value]
                self.slots[slot] = value
            if len(turn) >= 3:
                self.predicted_slots = {}
                for slot, value in turn[2].items():
                    slot = Slot(slot)
                    if isinstance(value, str):
                        value = [value]
                    self.predicted_slots[slot] = value

    def context(self):
        return self.dialogue.turns[:self.index+1]

    def dialogue_state(self):
        state = {}
        for turn in self.context():
            if turn.slots:
                state.update(
                    (slot, values) for slot, values in turn.slots.items()
                    if slot not in state or values is not None
                )
        return state

    def predicted_dialogue_state(self):
        state = {}
        for turn in self.context():
            if turn.predicted_slots:
                state.update(
                    (slot, values) for slot, values in turn.predicted_slots.items()
                    if slot not in state or values
                )
        return state

    def display(
        self,
        window=1,
        entire_state=False,
        include_description=False,
        include_empty_slots=True,
        examples_limit=0
    ):
        context = self.context()[max(0, self.index-window+1):self.index+1]
        display = '\n'.join(turn.turn for turn in context)
        if self.slots:
            lines = []
            if entire_state:
                slots = self.dialogue_state().items()
            else:
                slots = self.slots.items() if self.slots else []
            for slot, values in slots:
                values = values or []
                if not include_empty_slots and not values:
                    continue
                description = slot.description or []
                description = description if include_description else ''
                if description:
                    description = '\n    ' + description # noqa
                exs = ', '.join(
                    slot.values if examples_limit is None else slot.values[:examples_limit]
                ) if slot.values else ''
                if exs:
                    exs = '\n    (' + exs + ')'
                lines.append(
                    f"  {slot.name}: {', '.join(values)}{description}{exs}"
                )
            display += '\n\n' + '\n'.join(lines)
        if self.predicted_slots:
            display += '\n\n' + '\n'.join(
                f"  {slot.name}: {', '.join(values)}"
                for slot, values in (
                    self.predicted_dialogue_state().items()
                    if entire_state else self.predicted_slots.items()
                ) if values is not None
            )
        return display

    def __str__(self):
        return f'{type(self).__name__}({self.turn}{" | " + str(len(self.slots)) + " slots" if self.slots else ""})'
    __repr__ = __str__


dialoguelike: T.TypeAlias = T.Union[
    turnlike,
    list[turnlike],
    'Dialogue'
]
class Dialogue:
    Turn = Turn
    def __init__(self, turns:dialoguelike=None):
        if turns is None:
            turns = []
        self.turns:list[Turn] = getattr(self, 'turns', [])
        self.data: T.Optional['DstData'] = getattr(self, 'data', None)
        if isinstance(turns, Dialogue):
            for turn in turns.turns:
                self.add(turn)
        elif isinstance(turns, list):
            for turn in turns:
                self.add(turn)
        else:
            self.add(turns)
    update = __init__

    def add(self, turnlike):
        self.turns.append(type(self).Turn(turnlike, dialogue=self, index=len(self.turns)))

    def domains(self):
        domains = set()
        for turn in self.turns:
            if turn.slots:
                for slot in turn.slots:
                    domains.add(slot.domain)
        return domains

    def __str__(self):
        unique_slots = set()
        for turn in self.turns:
            if turn.slots:
                unique_slots.update(slot.name for slot in turn.slots)
        return f'{type(self).__name__}({len(self.turns)} turns{", "+str(len(unique_slots))+" slots" if unique_slots else ""})'
    __repr__ = __str__


dstlike: T.TypeAlias = T.Union[
    turnlike,
    list[turnlike|Dialogue],
    list[list[turnlike]],
    'DstData'
]
class DstData(ez.Data):
    Dialogue = Dialogue
    def __init__(self, dialogues:dstlike, ontology:ontologylike=None, file:ez.filelike=None):
        self.dialogues:list[Dialogue] = getattr(self, 'dialogues', [])
        self.ontology:Ontology = (
            Ontology(ontology) if ontology else getattr(self, 'ontology', Ontology())
        )
        if isinstance(dialogues, DstData):
            file = dialogues.file if file is None else file
            for dialogue in dialogues.dialogues:
                self.add(dialogue)
        elif isinstance(dialogues, list):
            for dialogue in dialogues:
                self.add(dialogue)
        else:
            self.add(dialogues)
        self._file = file
        self.__post_init__()
    update = __init__

    def domains(self) -> dict[str, set[Dialogue]]:
        domains = {}
        for dialogue in self.dialogues:
            for turn in dialogue.turns:
                if turn.slots:
                    for slot in turn.slots:
                        domains.setdefault(slot.domain, set()).add(dialogue)
        return domains

    def state_update_accuracy(self, results: DstResults):
        correct_slots = []
        incorrect_slots = []
        correct_turns = []
        incorrect_turns = []
        for dialogue in self.dialogues:
            for turn in (turn for turn in dialogue.turns if turn.slots is not None):
                predicted_state_update = turn.predicted_slots or {}
                predicted_slots = {slot.name: values for slot, values in predicted_state_update.items()}
                label_state_update = turn.slots or {}
                label_slots = {slot.name: values for slot, values in label_state_update.items()}
                for slot, values in list(predicted_slots.items()):
                    if values is not None and all(not value for value in values):
                        predicted_slots[slot] = None
                for slot, values in list(label_slots.items()):
                    if values is not None and all(not value for value in values):
                        label_slots[slot] = None
                all_slots_correct_in_turn = True
                for slot, values in label_slots.items():
                    prediction = predicted_slots.get(slot)
                    if set(prediction or []) == set(values or []):
                        correct_slots.append((turn, slot))
                    else:
                        incorrect_slots.append((turn, slot))
                        all_slots_correct_in_turn = False
                if all_slots_correct_in_turn:
                    correct_turns.append(turn)
                else:
                    incorrect_turns.append(turn)
        slot_denominator = len(correct_slots) + len(incorrect_slots)
        results.slot_update_accuracy = len(correct_slots) / slot_denominator
        results.state_update_accuracy = len(correct_turns) / (
            len(correct_turns) + len(incorrect_turns)
        )
        return correct_slots, incorrect_slots, correct_turns, incorrect_turns

    def joint_goal_accuracy(self, results:DstResults):
        """
        Calculate slot accuracy and joint goal accuracy here
        """
        correct_turns = []
        incorrect_turns = []
        correct_slots = []
        incorrect_slots = []
        for dialogue in self.dialogues:
            for turn in (turn for turn in dialogue.turns if turn.slots is not None):
                state = turn.dialogue_state() # the (whole) dialogue state label for this turn
                predicted_state = turn.predicted_dialogue_state() # the (whole) dialogue state prediction for this turn
                label_slots = {slot.name: values for slot, values in state.items()}
                predicted_slots = {slot.name: values for slot, values in predicted_state.items()}
                for slot, values in list(predicted_slots.items()):
                    if values is not None and all(not value for value in values):
                        predicted_slots[slot] = None
                for slot, values in list(label_slots.items()):
                    if values is not None and all(not value for value in values):
                        label_slots[slot] = None
                all_slots_correct_in_turn = True
                for slot, values in label_slots.items():
                    prediction = predicted_slots.get(slot)
                    if set(prediction or []) == set(values or []):
                        correct_slots.append((turn, slot))
                    else:
                        incorrect_slots.append((turn, slot))
                        all_slots_correct_in_turn = False
                if all_slots_correct_in_turn:
                    correct_turns.append(turn)
                else:
                    incorrect_turns.append(turn)
        results.slot_accuracy = len(correct_slots) / (len(correct_slots) + len(incorrect_slots))
        results.joint_goal_accuracy = len(correct_turns) / (len(correct_turns) + len(incorrect_turns))
        return correct_slots, incorrect_slots, correct_turns, incorrect_turns

    def slot_update_f1(self, results:DstResults):
        """
        Calculate slot update f1 here
        """
        for dialogue in self.dialogues:
            for turn in (turn for turn in dialogue.turns if turn.slots is not None):
                state_update = turn.predicted_slots  # the dialogue state update label for this turn
                predicted_state_update = turn.predicted_dialogue_state()  # the state update prediction for this turn
        ...
        results.slot_update_f1 = ...

    def slot_correction_f1(self, results:DstResults):
        """
        Calculate slot correction f1 here
        """
        for dialogue in self.dialogues:
            for turn in (turn for turn in dialogue.turns if turn.slots is not None):
                state_update = turn.predicted_slots  # the dialogue state update label for this turn
                predicted_state_update = turn.predicted_dialogue_state()  # the state update prediction for this turn
        ...
        results.slot_correction_f1 = ...

    def average_joint_goal_accuracy(self, results:DstResults):
        """
        Calculate average joint goal accuracy here
        """
        for dialogue in self.dialogues:
            for turn in (turn for turn in dialogue.turns if turn.slots is not None):
                state = turn.dialogue_state()  # the (whole) dialogue state label for this turn
                predicted_state = turn.predicted_dialogue_state()  # the (whole) dialogue state prediction for this turn
        ...
        results.average_joint_goal_accuracy = ...

    def flexible_goal_accuracy(self, results:DstResults):
        """
        Calculate flexible goal accuracy here
        """
        ...
        results.flexible_goal_accuracy = ...


    def state_update_similarity_score(self, results:DstResults, score_fn=None):
        return
        if score_fn is None: # noqa
            def score_with_bert(candidates, references):
                batch_size = 128
                batches = []
                for i in range(0, len(candidates), batch_size):
                    batch = (candidates[i:i+batch_size], references[i:i+batch_size])
                    batches.append(batch)
                results = []
                for cands, refs in batches:
                    p, r, f1 = bertscore(
                        cands=cands,
                        refs=refs,
                        lang='en',
                        verbose=False,
                        rescale_with_baseline=True,
                    )
                    results.append(f1)
                return [item.item() for sublist in results for item in sublist]
            score_fn = score_with_bert
        turn_slot_score_weight = []
        def calculate_baseline_value_similarities():
            """Calculate the baseline value similarity score for each slot type using ontology example values"""
            baseline_value_similarities = {}
            slots_missing_examples = []
            to_calc = []
            for slot in self.ontology.slots():
                examples = slot.values
                if examples:
                    for value_a, value_b in itertools.pairwise(examples):
                        to_calc.append((slot, f'{slot.name}: {value_a}', f'{slot.name}: {value_b}'))
                else:
                    slots_missing_examples.append(slot)
            if to_calc:
                slots, cands, refs = list(zip(*to_calc))
                calcs = score_fn(cands, refs)
                calc_by_slot = {}
                for slot, calc in zip(slots, calcs):
                    calc_by_slot.setdefault(slot, []).append(calc)
                for slot, calcs in calc_by_slot.items():
                    baseline_value_similarities[slot] = sum(calcs) / len(calcs)
            if baseline_value_similarities:
                prior_baseline = sum(baseline_value_similarities.values()) / len(baseline_value_similarities)
            else:
                prior_baseline = 0.0
            for slot in slots_missing_examples:
                baseline_value_similarities[slot] = prior_baseline
            return baseline_value_similarities, prior_baseline
        baseline_value_similarities, prior_baseline_value_similarity = calculate_baseline_value_similarities()
        def calculate_baseline_slot_similarities():
            """Calculate the baseline slot similarity for each dialogue domain"""
            baseline_slot_similarities = {}
            to_calc = []
            for domain, slots in self.ontology.domains().items():
                for slot_a, slot_b in itertools.pairwise(slots):
                    to_calc.append((domain, slot_a.name, slot_b.name))
            if to_calc:
                domains, slot_as, slot_bs = list(zip(*to_calc))
                calcs = score_fn(slot_as, slot_bs)
                calc_by_domain = {}
                for domain, calc in zip(domains, calcs):
                    calc_by_domain.setdefault(domain, []).append(calc)
                for domain, calcs in calc_by_domain.items():
                    baseline_slot_similarities[domain] = sum(calcs) / len(calcs)
            if baseline_slot_similarities:
                prior_baseline = sum(baseline_slot_similarities.values()) / len(baseline_slot_similarities)
            else:
                prior_baseline = 0.0
            return baseline_slot_similarities, prior_baseline
        baseline_slot_similarities, prior_baseline_slot_similarity = calculate_baseline_slot_similarities()
        for dialogue in self.dialogues:
            for turn in (turn for turn in dialogue.turns if turn.slots is not None):
                lslots: dict[Slot, list[str]] = turn.slots
                pslots: dict[Slot, list[str]] = turn.predicted_slots or {}
                def calculate_slot_pair_score_table():
                    slot_pair_score_table = [[0.0]*len(pslots) for _ in range(len(lslots))]  # lslot x pslot
                    to_calc = []
                    for l, (lslot, lvalues) in enumerate(lslots.items()):
                        for p, (pslot, pvalues) in enumerate((s, v) for s, v in pslots.items() if v is not None):
                            to_calc.append((l, p, lslot.name, pslot.name))
                    if to_calc:
                        ls, ps, lslot_names, pslot_names = list(zip(*to_calc))
                        calcs = score_fn(lslot_names, pslot_names)
                        for l, p, calc in zip(ls, ps, calcs):
                            slot_pair_score_table[l][p] = calc
                    return slot_pair_score_table
                slot_pair_score_table = calculate_slot_pair_score_table()
                def calculate_slot_value_pair_score_table():
                    slotvalue_pair_score_table = [[0.0]*len(pslots) for _ in range(len(lslots))] # lslot x pslot
                    empty_lslots = set()
                    empty_pslots = set()
                    to_calc = []
                    for l, (lslot, lvalues) in enumerate(lslots.items()):
                        if lvalues is None:
                            empty_lslots.add(lslot)
                            continue
                        for p, (pslot, pvalues) in enumerate(pslots.items()):
                            if pvalues is None:
                                empty_pslots.add(pslot)
                            else:
                                to_calc.append((l, p,
                                    f'{lslot.name}: {", ".join(lvalues)}',
                                    f'{pslot.name}: {", ".join(pvalues)}'
                                ))
                    if to_calc:
                        ls, ps, lslot_values, pslot_values = list(zip(*to_calc))
                        calcs = score_fn(lslot_values, pslot_values)
                        for l, p, calc in zip(ls, ps, calcs):
                            slotvalue_pair_score_table[l][p] = calc
                    return slotvalue_pair_score_table, empty_lslots, empty_pslots
                slot_value_pair_score_table, empty_lslots, empty_pslots = calculate_slot_value_pair_score_table()
                def adjust_slot_pair_score_table():
                    for l, lslot in enumerate(lslots):
                        slot_pair_row = slot_pair_score_table[l]
                        baseline = baseline_slot_similarities.get(lslot.domain, prior_baseline_slot_similarity)
                        for p, score in enumerate(slot_pair_row):
                            slot_pair_row[p] = (score - baseline) / (1.0 - baseline)
                adjust_slot_pair_score_table()
                def adjust_slot_value_pair_score_table():
                    for l, lslot in enumerate(lslots):
                        if lslot in empty_lslots:
                            continue
                        slotvalue_pair_row = slot_value_pair_score_table[l]
                        for p, score in enumerate(slotvalue_pair_row):
                            baseline = baseline_value_similarities.get(lslot, prior_baseline_value_similarity)
                            slotvalue_pair_row[p] = (score - baseline) / (1.0 - baseline)
                adjust_slot_value_pair_score_table()
                def calculate_scores_and_weights():
                    tssw = []
                    for l, (slot, slot_pair_row, slotvalue_pair_row) in enumerate(zip(
                        lslots, slot_pair_score_table, slot_value_pair_score_table
                    )):
                        if slot in empty_lslots:
                            weight = max(slot_pair_row) if slot_pair_row else 0.0
                            tssw.append((turn, slot, 0.0, max(0, weight)))
                        else:
                            if not slot_pair_row:
                                tssw.append((turn, slot, 0.0, 0.0))
                            else:
                                softmax_denominator = sum(math.exp(score) for score in slot_pair_row)
                                softmax = [math.exp(score) / softmax_denominator for score in slot_pair_row]
                                score = sum(softmax[i] * slotvalue_pair_row[i] for i in range(len(softmax)))
                                tssw.append((turn, slot, score, 1.0))
                    return tssw
                tssw = calculate_scores_and_weights()
                turn_slot_score_weight.extend(tssw)
        numerator = sum(score * weight for _, _, score, weight in turn_slot_score_weight)
        denominator = sum(weight for _, _, _, weight in turn_slot_score_weight)
        results.state_update_similarity_score = numerator / denominator
        return turn_slot_score_weight

    def add(self, dialogue:dialoguelike):
        dialogue = type(self).Dialogue(dialogue)
        dialogue.data = self
        self.dialogues.append(dialogue)
        for turn in dialogue.turns:
            if turn.slots:
                for slot, values in list(turn.slots.items()):
                    del turn.slots[slot]
                    slot = self.ontology.add(slot)
                    turn.slots[slot] = values
            if turn.predicted_slots:
                for slot, values in list(turn.predicted_slots.items()):
                    del turn.predicted_slots[slot]
                    slot = self.ontology.add(slot)
                    turn.predicted_slots[slot] = values
        return dialogue

    def __str__(self):
        return f'{type(self).__name__}({len(self.dialogues)} dialogues{" from "+str(self.file) if self.file else ""})'
    __repr__ = __str__



@dataclasses.dataclass
class DstResults(Results):
    slot_accuracy: float = None
    """The proportion of state slots where the predicted value matches the label value"""
    joint_goal_accuracy: float = None
    """The proportion of turns where all slots in the predicted state match the label state"""
    slot_update_f1: float = None
    """The f1 score of predicted slot updates matching label slot updates"""
    state_update_accuracy: float = None
    """The proportion of turns where the entire state update matches the label state update"""
    slot_update_accuracy: float = None
    """The proportion of state slot updates where the predicted value update matches the label value update"""
    slot_correction_f1: float = None
    """The f1 score of predicted slot updates matching the difference between the label state previous turn's predicted state"""
    average_joint_goal_accuracy: float = None
    """The proportion of turns where all filled predicted slots match all filled label slots"""
    flexible_goal_accuracy: float = None
    """A version of JGA that gives lower weight to slots updated by more historical turns"""
    state_similarity_score: float = None
    """The average similarity of predicted dialogue states to label dialogue states, using sequence similarity metrics like BERTscore"""
    state_update_similarity_score: float = None
    """The average similarity of predicted dialogue state updates to label dialogue state updates, using sequence similarity metrics like BERTscore"""
    slot_similarity_score: float = None
    """The average similarity of predicted slot values to label slot values, using a sequence similarity metric like BERTscore"""
    good_slots: list[tuple[Turn, Slot]] = None
    bad_slots: list[tuple[Turn, Slot]] = None
    good_turns: list[Turn] = None
    bad_turns: list[Turn] = None
















