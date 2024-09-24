
import ezpyz as ez
from dst.approaches.t5 import T5
import dst.analysis.experiments_folder as exp
import dst.data.dst_data as dst
import dst.approaches.dst_seq_data as dstseq
from dst.approaches.seq2seq_dst import Seq2seqDst
from dst.approaches.seq2seq_dsg import Seq2seqDSG
import json
import pathlib as pl
import random
from dst.analysis.hand_test import get_model_path_and_type
from tqdm import tqdm


def group(items, key, sort=None):
    if callable(key):
        key = [key(item) for item in items]
    if not isinstance(key, list):
        key = [item[key] for item in items]
    assert len(key) == len(items), \
        "Length of 'key' list must match the length of 'items'"
    if sort is not None:
        if callable(sort):
            sort = {id(item): sort(item) for item in items}
        elif isinstance(sort, list):
            assert len(sort) == len(items), \
                "Length of 'sort' list must match the length of 'items'"
            sort = {id(item): sort[i] for i, item in enumerate(items)}
        else:
            sort = {id(item): item[sort] for item in items}
    grouped_dict = {group: [] for group in set(key)}
    for item, group in zip(items, key):
        grouped_dict[group].append(item)
    if isinstance(sort, dict):
        for group in grouped_dict:
            sortable = sorted((sort[id(item)], i, item) for i, item in enumerate(grouped_dict[group]))
            grouped = [item for _, _, item in sortable]
            grouped_dict[group] = grouped
    return grouped_dict


def predict_to_table(data_path, model_path):
    data_path = pl.Path(data_path)
    turn_table_path = data_path / 'turn.csv'
    data_raw = ez.File(turn_table_path).load()
    columns = data_raw[0]
    data = [[json.loads(cell) for cell in row] for row in data_raw[1:]]
    turns = []
    for row in data:
        turns.append(dict(zip(columns, row)))
    del turn_table_path, columns, data, row
    dialogues = group(turns, 'dialogue', 'turn_index')
    dsg = Seq2seqDSG.load(model_path)
    slot_value_table = [['slot', 'value', 'turn_id', 'slot_id', 'slot_value_id']]
    slot_table = [['slot', 'domain', 'description', 'slot_id']]
    value_candidate_table = [['candidate_value', 'slot_id', 'is_provided', 'value_candidate_id']]
    for dialouge_id, dialogue_turns in tqdm(list(dialogues.items()), desc='Predicting'):
        dialogue_turn_texts = [t['text'] for t in dialogue_turns]
        dialogue_turn_ids = [t['turn_id'] for t in dialogue_turns]
        dialogue_turn_domains = [t['domain'] for t in dialogue_turns]
        dialogue_obj = dstseq.DstSeqData(dialogues=[dialogue_turn_texts])
        predictions = dsg.predict(dialogue_obj)
        state_updates = [
            turn.predicted_slots for dialogue in predictions.dialogues for turn in dialogue.turns]
        for turn_id, domain, state_update in zip(dialogue_turn_ids, dialogue_turn_domains, state_updates):
            for slot_obj, slot_value in state_update.items():
                slot_name = slot_obj.name
                slot_id = ez.uuid()
                slot_value_id = ez.uuid()
                value_candidate_id = ez.uuid()
                slot_value_row = [json.dumps(x) for x in (
                    slot_name, slot_value, turn_id, slot_id, slot_value_id
                )]
                slot_row = [json.dumps(x) for x in (
                    slot_name, domain, '', slot_id
                )]
                value_candidate_row = [json.dumps(x) for x in (
                    slot_value, slot_id, False, value_candidate_id
                )]
                slot_value_table.append(slot_value_row)
                slot_table.append(slot_row)
                value_candidate_table.append(value_candidate_row)
    data_save_path = data_path.with_name(f"{data_path.name}_DSG")
    ez.File(data_save_path / 'slot_value.csv').save(slot_value_table)
    ez.File(data_save_path / 'slot.csv').save(slot_table)
    ez.File(data_save_path / 'value_candidate.csv').save(value_candidate_table)
    ez.File(data_save_path / 'turn.csv').save(data_raw)



if __name__ == '__main__':
    task = 'Seq2seqDSG'
    experiment = 'DaringWookiee'
    iteration = 1
    data = 'mwoz2.1'
    #data = 'sgd_wo_domains'
    predict_to_table(
        f'data/{data}/valid',
        f'ex/{task}/{experiment}/{iteration}/model')