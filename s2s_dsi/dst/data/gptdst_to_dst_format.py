import math

from dst.data.gpt_generate_data import Example, Turn, Dialogue, Slot
import dst.data.dst_data as dst
from dst.data.split import random_split, leave_n_out_splits

import pickle
import random
import json


def gpt_generated_to_dst_examples(dialogues: list[Dialogue]) -> dst.DstData:
    degenerate = []
    domains = {}
    ds = []
    for dialogue in dialogues:
        ts = []
        slot_coordinator = {}
        for i, turn in enumerate(dialogue.turns):
            example = turn.example
            ss = {}
            for slot_name, slot in example.slots.items():
                slot_name = postprocess_slot_name(slot_name)
                if slot.value is None:
                    degenerate.append((turn, slot))
                    continue
                values = []
                for value in slot.alternatives:
                    value = postprocess_slot_value(value, turn.speaker, turn.listener, slot.answer)
                    values.extend(value)
                if slot_name in slot_coordinator:
                    s = slot_coordinator[slot_name] # noqa
                    s.domain = dialogue.scenario
                    s.description = slot.description
                    s.values = values
                else:
                    s = dst.Slot(
                        name=slot_name,
                        domain=dialogue.scenario,
                        description=slot.description,
                        values=values,
                        categorical=False
                    )
                    domains.setdefault(s.domain, []).append(s)
                if s.description is None:
                    slot_coordinator[s.name] = s
                slot_value = postprocess_slot_value(
                    slot.value, turn.speaker, turn.listener, slot.answer
                )
                ss[s] = slot_value
            t = dst.Turn(
                turn=turn.text,
                speaker=turn.speaker,
                listener=turn.listener,
            )
            ts.append((t, ss))
        ts = [dst.Turn(turn=t, slots=ss) for t, ss in ts]
        d = dst.Dialogue(ts)
        ds.append(d)
    ontology = [slot for domain in domains.values() for slot in domain]
    data = dst.DstData(dialogues=ds, ontology=ontology)
    return data

def add_negatives(
    data: dst.DstData,
    num_negatives:int|tuple[int,int]|str|tuple[str,int]=('match', 5)
):
    domains = data.domains()
    for domain, dialogues in domains.items():
        for dialogue in dialogues:
            candidates = [
                slot
                for d in dialogues
                for t in d.turns
                for slot in t.slots
                if d is not dialogue
            ]
            for turn in dialogue.turns:
                turn_candidates = [
                    slot for slot in candidates
                    if not any(slot.name == s.name for s in turn.slots)
                ]
                if num_negatives == 'match':
                    num_negatives = len(turn.slots)
                elif isinstance(num_negatives, tuple) and isinstance(num_negatives[0], str):
                    num_negatives = max(num_negatives[1], len(turn.slots))
                if isinstance(num_negatives, tuple) and isinstance(num_negatives[0], int):
                    num_to_add = random.randint(*num_negatives)
                else:
                    num_to_add = 0 if num_negatives is None else num_negatives
                negatives = random.sample(
                    turn_candidates, min(num_to_add, len(turn_candidates))
                )
                for slot in negatives:
                    turn.slots[slot] = None # noqa

def camel_case_to_text(s):
    new = ''
    for i, ch in enumerate(s):
        if i > 0 and s[i-1].islower() and ch.isupper():
            new += ' ' + ch.lower()
        else:
            new += ch
    return new

def snake_case_to_text(s):
    return ' '.join(s.split('_'))

def postprocess_slot_name(slot_name:str):
    slot_name = camel_case_to_text(slot_name)
    slot_name = snake_case_to_text(slot_name)
    return slot_name

def split_on(text, chars):
    parts = []
    i = 0
    for j, c in enumerate(text):
        if c in chars:
            parts.append(text[i:j])
            i = j + 1
    parts.append(text[i:])
    return parts

def postprocess_slot_value(value: str, speaker: str, listener: str, answer: str) -> list[str]:
    if value is None:
        return list()
    else:
        normalized = ''.join(c.lower() for c in value if c.isalnum())
        if normalized in {'me', 'i'}:
            return ['speaker']
        elif normalized in {'you'}:
            return ['listener']
        else:
            parts = split_on(value, ',|;')
            values = []
            for part in parts:
                if part.startswith(' and'):
                    part = part[len(' and'):]
                if part.startswith(' or'):
                    part = part[len(' or'):]
                part = part.strip()
                if part:
                    values.append(part)
            values = [value for value in values if value.lower() not in {'etc', 'etc.', '...'}]
            return values


def load_examples(path, display_degenerate_dialogue_distribution=False):
    with open(path, 'rb') as file:
        examples: list[list[Example]] = pickle.load(file)
    dialogues = {}
    no_slots_distribution = {}
    for turns in examples:
        for turn_example in turns:
            turn_example: Example
            turn = turn_example.turn
            dialogue = turn.dialogue
            num_without_slots = sum(1 for turn in dialogue.turns if not turn.example.slots)
            dialogues[id(dialogue)] = dialogue
            no_slots_distribution.setdefault(num_without_slots, set()).add(id(dialogue))
    if display_degenerate_dialogue_distribution:
        for num_without_slots, ids in sorted(no_slots_distribution.items()):
            print(f'{num_without_slots} turns without slots: {len(ids)} dialogues')
    result = list(dialogues.values())
    return result


def get_qa_pairs_from_generated_dialogues(dialogues: list[Dialogue]):
    turn_slot_value_to_qa_pairs = {}
    degenerate = []
    domains = {}
    ds = []
    for dialogue in dialogues:
        ts = []
        slot_coordinator = {}
        for i, turn in enumerate(dialogue.turns):
            example = turn.example
            ss = {}
            for slot_name, slot in example.slots.items():
                slot_name = postprocess_slot_name(slot_name)
                if slot.value is None:
                    degenerate.append((turn, slot))
                    continue
                question = slot.question
                answer = slot.answer
                values = []
                for value in slot.alternatives:
                    value = postprocess_slot_value(value, turn.speaker, turn.listener, slot.answer)
                    values.extend(value)
                if slot_name in slot_coordinator:
                    s = slot_coordinator[slot_name] # noqa
                    s.domain = dialogue.scenario
                    s.description = slot.description
                    s.values = values
                else:
                    s = dst.Slot(
                        name=slot_name,
                        domain=dialogue.scenario,
                        description=slot.description,
                        values=values,
                        categorical=False
                    )
                    domains.setdefault(s.domain, []).append(s)
                if s.description is None:
                    slot_coordinator[s.name] = s
                slot_value = postprocess_slot_value(
                    slot.value, turn.speaker, turn.listener, slot.answer
                )
                ss[s] = slot_value
                turn_slot_value_to_qa_pairs.setdefault(
                    (turn.text, slot_name, ', '.join(slot_value)), []
                ).append((question, answer))
            t = dst.Turn(
                turn=turn.text,
                speaker=turn.speaker,
                listener=turn.listener,
            )
            ts.append((t, ss))
        ts = [dst.Turn(turn=t, slots=ss) for t, ss in ts]
        d = dst.Dialogue(ts)
        ds.append(d)
    ontology = [slot for domain in domains.values() for slot in domain]
    data = dst.DstData(dialogues=ds, ontology=ontology)
    tsv_to_qa = {}
    for (turn, slot, value), qas in turn_slot_value_to_qa_pairs.items():
        question, answer = qas[0]
        tsv_to_qa['|'.join((turn, slot, value))] = (question, answer)
        if len(set(qas)) != 1:
            print(f'Warning: multiple QA pairs for ({turn}, {slot}, {value}):')
            for question, answer in qas:
                print(f'  {question} -> {answer}')
    with open('data/gptdst5k/turn_slot_value_to_QA.json', 'w') as f:
        json.dump(tsv_to_qa, f, indent=2)
    return turn_slot_value_to_qa_pairs


def main():
    from ezpyz import explore as ex
    dialogues = load_examples('data/gptdst5k/examples.pkl')
    gptdst = gpt_generated_to_dst_examples(dialogues)
    turns = [turn for dialogue in gptdst.dialogues for turn in dialogue.turns]
    avg_num_slots = sum(len(turn.slots) for turn in turns) / len(turns)
    add_negatives(gptdst)
    train, valid, test = random_split(
        gptdst, train_size=0.9, valid_size=0.05, test_size=0.05, seed=42
    )
    by_domain_splits = leave_n_out_splits(
        gptdst, num_splits=10, valid_size=50, test_size=50, seed=42
    )
    # ex()
    gptdst.save('data/gptdst5k/gptdst5k.pkl')
    train.save('data/gptdst5k/gptdst5k_train.pkl')
    valid.save('data/gptdst5k/gptdst5k_valid.pkl')
    test.save('data/gptdst5k/gptdst5k_test.pkl')
    for i, (train, valid, test) in by_domain_splits.items():
        train.save(f'data/gptdst5k/gptdst5k_train_domains_{i}.pkl')
        valid.save(f'data/gptdst5k/gptdst5k_valid_domains_{i}.pkl')
        test.save(f'data/gptdst5k/gptdst5k_test_domains_{i}.pkl')


if __name__ == '__main__':
    get_qa_pairs_from_generated_dialogues(load_examples('data/gptdst5k/examples.pkl'))