
import random
import itertools
import functools
import ezpyz as ez
from ezpyz import explore as ex
import src.dst.data.dst_data as dst


def split(s, *splitters):
    if len(splitters) == 0:
        return [s]
    else:
        return functools.reduce(
            lambda acc, splitter: list(itertools.chain(*[a.split(splitter) for a in acc])), splitters, [s])

ex = ez.bind(ex)(max_col=100)

data = dst.DstData.load('data/sgd-data/sgd_100_valid.pkl')

num_dialogues = len(data.dialogues)
print('Number of dialogues:', num_dialogues)

num_turns = sum(len(d.turns) for d in data.dialogues)
print('Number of turns:', num_turns)

print('Number of turns per dialogue:', num_turns / num_dialogues)

num_tokens = sum(len(t.turn.split()) for d in data.dialogues for t in d.turns)
print('Number of tokens:', num_tokens)

print('Number of tokens per turn:', num_tokens / num_turns)

slots = []
slot_names = set()
domains = {}
boolean_slots = []
request_slots = []
listed_slots = []
dialogues = {}
turns = {}
for d in data.dialogues:
    for t in d.turns:
        if t.slots is not None:
            for s, v in t.slots.items():
                if v:
                    turns.setdefault(t, []).append((s, v))
                    dialogues.setdefault(d, []).append((s, v))
                    slots.append((s, v))
                    slot_names.add(s.name)
                    domains.setdefault(s.domain, set()).add(s.name)
                    if len(v) > 1:
                        listed_slots.append(s)
                    else:
                        v = v[0].lower()
                        if v in {'true', 'false', 'yes', 'no'}:
                            boolean_slots.append(s)
                        elif v == '?':
                            request_slots.append(s)

turns_valid_for_extraction = [turn for dialogue in data.dialogues for turn in dialogue.turns if turn.slots is not None]

print('Number of slot extractions', len(slots))
print('Number of unique slot names:', len(slot_names))
print('Number of domains:', len(domains))
print('Number of unique slot names per domain:', sum(len(v) for v in domains.values()) / len(domains))
print('Number of unique slot names per dialogue:', sum(len(v) for v in dialogues.values()) / len(dialogues))
if data.file.path.name.startswith('sgd'):
    print('Number of unique slot types per domain:', sum(len(v) for v in domains.values()) / len(domains) / 6)
    print('Number of unique slot types per dialogue:', sum(len(set(s.description for s, _ in v)) for v in dialogues.values()) / len(dialogues))
print('Number of slots per turn:', sum(len(v) for v in turns.values()) / len(turns_valid_for_extraction))
print('Number of turns with no slots:', num_turns - len(turns))

print('Number of boolean slots:', len(boolean_slots))
print('Number of request slots:', len(request_slots))
print('Number of listed slots:', len(listed_slots))


avg_slot_name_tokens = sum(len(split(s.name, ' ', '_')) for s, _ in slots) / len(slots)
print('Number of tokens per slot name:', avg_slot_name_tokens)

avg_slot_value_tokens = sum(len(split(", ".join(v), ' ', '_')) for _, v in slots) / len(slots)
print('Number of tokens per slot value:', avg_slot_value_tokens)







