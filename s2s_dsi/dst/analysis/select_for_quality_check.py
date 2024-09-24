

from dst.data.dst_data import DstData

from functools import reduce
from operator import add
import random

from ezpyz import File


def dialogues_to_csv(start=0, end=10):
    data = DstData.load('data/gptdst5k/gptdst5k_test_domains_0.pkl')
    dialogues = list(data.dialogues)
    random.seed(42)
    random.shuffle(dialogues)
    selected = dialogues[start:end]
    header = ['dialogue', 'turn', 'text', 'slot', 'value', 'description']
    rows = [header]
    for i, dialogue in enumerate(selected):
        index = i + start
        rows.append([str(index), '###', '######', '###', '###', '######'])
        for j, turn in enumerate(dialogue.turns):
            rows.append(['', str(j), turn.turn, '', '', ''])
            for k, (slot, value) in enumerate(turn.slots.items()):
                rows.append(['', '', '', slot.name, ', '.join(value or ()), slot.description])
    csv_file = File(f'data/quality_check/dialogues-{start}-{end}.csv')
    csv_file.save(rows)

if __name__ == '__main__':
    dialogues_to_csv(0, 10)


