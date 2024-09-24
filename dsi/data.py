
import dsi.dst_data as dst
import ezpyzy as ez
import pathlib as pl


def replace_mwoz_labels_with_dsg_silver_labels(data: dst.Data) -> dst.Data:
    original_valid_dials = ez.File('data/mwoz2.1/dev_dials.json').load()
    original_dialogue_ids = []
    original_dialogue_turns = []
    for dial in original_valid_dials:
        original_dialogue_ids.append(dial['dialogue_idx'])
        original_turns = []
        for turn in dial['dialogue']:
            original_turns.append(turn['system_transcript'])
            original_turns.append(turn['transcript'])
        original_dialogue_turns.append(original_turns)
    idx, dialogue_id, turn_index, turn_content, slot, value, context = list(range(7))
    silver = ez.File('data/silver_mwoz/mwoz_valid_svtcidx_pred.csv').load()
    silver = silver[1:]
    by_dialogue_turn = {} # (dialogue_idx, turn_idx) -> dict[slot, value]
    for row in silver:
        dialogue_idx = int(row[dialogue_id])
        turn_idx = int(row[turn_index]) + 1
        original_turn = original_dialogue_turns[dialogue_idx][turn_idx]
        silver_turn = row[turn_content]
        assert original_turn == silver_turn
        if (dialogue_idx, turn_idx) not in by_dialogue_turn:
            by_dialogue_turn[(dialogue_idx, turn_idx)] = {}
        by_dialogue_turn[(dialogue_idx, turn_idx)][row[slot]] = row[value]
    turns = ~data.turns
    did_tidx_to_tid = {}
    for did, tidx, tid in zip(turns.dialogue, turns.turn_index, turns.turn_id):
        did_tidx_to_tid[(did, tidx)] = tid
    slot_values = []
    slots = {}
    value_candidates = {}
    for (dialogue_idx, turn_idx), svs in by_dialogue_turn.items():
        did = original_dialogue_ids[dialogue_idx]
        tid = did_tidx_to_tid[(did, turn_idx)]
        domain = turns[tid].domain()
        for s, v in svs.items():
            slot_row = slots.setdefault(s, dict(slot=s, domain=domain, slot_id=ez.uuid()))
            sid = slot_row['slot_id']
            slot_value = dict(
                slot=s, value=v, turn_id=tid, slot_id=sid, slot_value_id=ez.uuid())
            value_candidates.setdefault(s, []).append(v)
            slot_values.append(slot_value)
    value_candidate_rows = []
    for s, vs in value_candidates.items():
        for v in vs:
            sid = slots[s]['slot_id']
            value_candidate_rows.append(dict(
                candidate_value=v, slot_id=sid, value_candidate_id=ez.uuid()))
    slot_value_table = dst.SlotValue.of(slot_values)
    slot_table = dst.Slot.of(list(slots.values()))
    value_candidate_table = dst.ValueCandidate.of(value_candidate_rows)
    silver_data = dst.Data()
    silver_data.turns = turns
    silver_data.slot_values = slot_value_table
    silver_data.slots = slot_table
    silver_data.value_candidates = value_candidate_table
    return silver_data

def fix_silver_sgd_files(folder):
    folder = pl.Path(folder)
    slot_value_path = folder/'slot_value.csv'
    slot_values_table = ez.Table.of(slot_value_path)
    slot_values_table.value[:] = [
        ', '.join(value) for value in slot_values_table.value
    ]
    value_cand_path = folder/'value_candidate.csv'
    value_candidate_table = ez.Table.of(value_cand_path)
    value_candidate_table.candidate_value[:] = [
        ', '.join(value) for value in value_candidate_table.candidate_value
    ]
    slot_values_table().save(slot_value_path)
    value_candidate_table().save(value_cand_path)



if __name__ == '__main__':
    # silver = dst.Data('data/silver_mwoz/valid')
    # exs:ez.Table = silver.slot_values.turn_id << silver.turns.turn_id
    # exs = exs[exs.text, exs.slot, exs.value][:100]
    # print(exs().display(max_cell_width=80))

    # fix_silver_sgd_files('data/silver_sgd/valid')

    silver = dst.Data('data/silver_sgd/valid')
    ...