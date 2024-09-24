
import ezpyz as ez
from dst.approaches.t5 import T5
import dst.analysis.experiments_folder as exp
import dst.data.dst_data as dst
import dst.approaches.dst_seq_data as dstseq
from dst.approaches.seq2seq_dst import Seq2seqDst
from dst.approaches.seq2seq_dsg import Seq2seqDSG
import random


def sample_examples(
    experiment, iteration, data, split='valid', task='Seq2seqDst', n=1, mode='good and bad', show_entire_state=False
):
    experiments = exp.ExperimentsFolder()[task]
    experiment = experiments[experiment]
    iteration = experiment[str(iteration)]
    result = iteration[data]
    print('\n'.join(f'{str(k):.30}: {str(v):.70}' for k, v in result.dict().items()), '\n\n')
    predictions = result.predictions(split)
    data:dst.DstData = predictions.load()
    if mode == 'good and bad':
        new_results = dst.DstResults()
        good_slots, bad_slots, good_turns, bad_turns = data.state_update_accuracy(new_results)
        good_candidates = [
            turn for turn in good_turns if any(value for key, value in turn.slots.items())
        ]
        bad_candidates = [
            turn for turn in bad_turns if any(value for key, value in turn.slots.items())
        ]
        good = random.sample(good_candidates, min(n, len(good_candidates)))
        print('Good Turns'.center(80, '-'), '\n')
        for turn in good:
            print(turn.display(
                10, entire_state=show_entire_state, include_description=False, examples_limit=None
            ))
            print('\n')
        bad = random.sample(bad_candidates, min(n, len(bad_candidates)))
        print('Bad Turns'.center(80, '-'), '\n')
        for turn in bad:
            print(turn.display(10))
            print('\n')
        return
    elif mode == 'sequences':
        turn_slots = []
        for dialogue in data.dialogues:
            for turn in dialogue.turns:
                if turn.slots is not None:
                    turn_slots.append((turn, turn.slots))
        sample = random.sample(turn_slots, min(n, len(turn_slots)))
        for turn, slots in sample:
            ...
    else:
        dialogues = random.sample(data.dialogues, min(n, len(data.dialogues)))
        for dialogue in dialogues:
            print('-' * 80)
            for turn in (turn for turn in dialogue.turns if turn.slots is not None):
                print(turn.display(
                    2, entire_state=show_entire_state, include_description=False,
                    include_empty_slots=False
                ), '\n')
        return


def get_model_path_and_type(experiment, iteration, task='Seq2seqDst'):
    experiments = exp.ExperimentsFolder()[task]
    experiment = experiments[experiment]
    iteration = experiment[str(iteration)]
    model_path = iteration.model_path
    model_type = dict(Seq2seqDst=Seq2seqDst, Seq2seqDSG=Seq2seqDSG)[task]
    return model_path, model_type


def hand_test(model, model_type, inputs:list[str]=None):
    iter_input = iter(inputs) if inputs is not None else None
    dialogue = []
    slots = []
    try:
        while True:
            dialogue = get_dialogue(dialogue) if inputs is None else [next(iter_input)]
            if model_type is Seq2seqDst:
                model: Seq2seqDst
                slots = get_slots(slots)
                data = dstseq.DstSeqData([[
                    (turn, dict.fromkeys(slots))
                    for turn in dialogue
                ]])
                prediction = model.predict(data)

            else:
                data = dstseq.DstSeqData([dialogue])
                prediction = model.predict(data)
            print()
            for d in prediction.dialogues:
                print('   ', d.turns[-1].display(1, entire_state=False, include_description=False), '\n')
    except StopIteration:
        return


def get_dialogue(previous_dialogue=None):
    if previous_dialogue is None:
        previous_dialogue = []
    turn = input('Dialogue: ')
    if not turn:
        return previous_dialogue
    else:
        dialogue = []
        speaker = 'B'
        while turn:
            dialogue.append(turn)
            turn = input(f'{speaker}: ')
            speaker = 'A' if speaker == 'B' else 'B'
        return dialogue

def get_slots(previous_slots=None):
    if previous_slots is None:
        previous_slots = []
    slot = input('Slot: ')
    if not slot:
        return previous_slots
    slots = []
    while slot:
        slot = tuple([x.strip() for x in slot.split(':', 1)])
        slot += (None,) * (2 - len(slot))
        slots.append(slot)
        slot = input('Slot: ')
    return slots


if __name__ == '__main__':
    task = 'Seq2seqDSG'
    experiment = 'DaringWookiee'
    iteration = 1
    # sample_examples(
    #     experiment,
    #     iteration,
    #     'gptdst5k_valid_domains_0',
    #     split='valid',
    #     n=10,
    #     mode='dialogues',
    #     show_entire_state=False,
    #     task=task
    # )
    conversation_text = '''S1: I do not have insurance to cover medical expenses. Where can I get help with paying for HIV care?'''
    conversation = [x.split(':', 1)[1].strip() for x in conversation_text.split('\n') if x.strip()]



    model_path, model_type = get_model_path_and_type(experiment, iteration, task=task)
    model: Seq2seqDSG = model_type.load(model_path, load_in_8bit=True)
    t5: T5 = model.seq2seq_model # noqa
    t5.generation_config.num_beams = 20
    t5.generation_config.repetition_penalty = 2.0
    data = dstseq.DstSeqData(dialogues=[conversation])
    predictions = model.predict(data)
    for dialogue in predictions.dialogues:
        for turn in dialogue.turns:
            print(turn.turn)
            print({slot.name: ', '.join(values) for slot, values in turn.predicted_slots.items()})
            print()
