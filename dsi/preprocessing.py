import math

import ezpyzy as ez
import dsi.dst_data as dst
import random as rng
import dsi.utils as utils

def downsample_examples(data, k):
    domains_to_svids = {}
    exs = data.slot_values.turn_id << data.turns.turn_id
    for domain, svid in zip(exs.domain, exs.slot_value_id):
        if domain not in domains_to_svids:
            domains_to_svids[domain] = []
        domains_to_svids[domain].append(svid)
    for svids in domains_to_svids.values():
        rng.shuffle(svids)
    interleaved_svids = list(utils.roundrobin(*domains_to_svids.values()))
    slot_value_id_sample = interleaved_svids[:k]
    data.slot_values = ~data.slot_values[slot_value_id_sample]
    if data.predictions:
        data.predictions = ~data.predictions[slot_value_id_sample]
    return data

def downsample_dialogues(data, k):
    examples = data.predictions.slot_value_id << data.slot_values.slot_value_id
    dialogues_examples = examples.turn_id << data.turns.turn_id
    dialogues_with_exs = set(dialogues_examples.dialogue)
    dialogue_sample = rng.sample(list(dialogues_with_exs), k=k)
    slots_values_with_dialogues = data.slot_values.turn_id << data.turns.turn_id
    in_sample_mask = [
        slot_value_id for slot_value_id, d in
        zip(slots_values_with_dialogues.slot_value_id, slots_values_with_dialogues.dialogue)
        if d in dialogue_sample
    ]
    data.slot_values = ~data.slot_values[in_sample_mask]
    data.predictions = ~data.predictions[in_sample_mask]
    return data


def exclude_speakers(data, speakers):
    slots_values_with_dialogues = data.slot_values.turn_id << data.turns.turn_id
    speaker_mask = [
        slot_value_id for slot_value_id, s in
        zip(slots_values_with_dialogues.slot_value_id, slots_values_with_dialogues.speaker)
        if s not in speakers
    ]
    data.slot_values = ~data.slot_values[speaker_mask]
    if data.predictions:
        data.predictions = ~data.predictions[speaker_mask]

def add_neg_slot_targets(data, per_domain=True, exclude_speakers=None):
    if exclude_speakers is None:
        exclude_speakers = set()
    if per_domain:
        train_turn_by_slot = data.turns.domain & data.slots.domain  # assign every slot to a turn based on all domains of positive examples for that turn
        speaker_mask = [x not in exclude_speakers for x in train_turn_by_slot.speaker]
        train_turn_by_slot = ~train_turn_by_slot[speaker_mask]
    else:
        valid_domains = set(data.slots.domain)
        domain_mask = [x in valid_domains for x in data.turns.domain]
        train_turn_by_slot = ~data.turns[domain_mask]
        speaker_mask = [x not in exclude_speakers for x in train_turn_by_slot.speaker]
        train_turn_by_slot = ~train_turn_by_slot[speaker_mask]
        train_turn_by_slot = train_turn_by_slot @ data.slots
    train_targets = (
        train_turn_by_slot[train_turn_by_slot.turn_id, train_turn_by_slot.slot_id] <<
        data.slot_values[data.slot_values.turn_id, data.slot_values.slot_id]
    )
    slot_values = train_targets().cast(dst.SlotValue)
    slot_values.slot_value_id = ez.IDColumn(slot_values.slot_value_id)
    data.slot_values = slot_values
    preds = dst.Prediction.of(dict(
        slot_value_id=slot_values.slot_value_id,
        slot_id=slot_values.slot_id,
    ), fill=None)
    data.predictions = preds

def drop_domains(data, domains: list[str], include_specified=False):
    domains = set(domains)
    slot_domains = set(data.slots.domain)
    if not include_specified and (not domains & slot_domains):
        return
    sv = data.slot_values
    slot_value_del = (~sv[sv.slot_id, sv.slot_value_id]).slot_id << data.slots.slot_id
    slot_value_del_mask = [
        (x.domain() in domains) != include_specified for x in slot_value_del
    ]
    del data.slot_values[slot_value_del_mask]
    if data.predictions:
        pred_del = data.predictions.slot_value_id << slot_value_del.slot_value_id
        pred_del_mask = [
            (x.domain() in domains) != include_specified for x in pred_del
        ]
        del data.predictions[pred_del_mask]
    ontology_del_mask = [
        (x.domain() in domains) != include_specified for x in data.slots
    ]
    ontology = dst.Slot.of({**data.slots().column_names, 'del_mask': ontology_del_mask})
    ontology_candidates = data.value_candidates.slot_id << ontology.slot_id
    del data.value_candidates[ontology_candidates.del_mask]
    del data.slots[ontology_del_mask]
    return

def add_continuation_values(
    data, pcent_of_continuation=1.0, pcent_of_existing=None, req_token=None
):
    data: dst.Data
    turns = data.turns
    original_pred_len = len(data.predictions)
    turn_location = {}  # turn_id -> (domain, dialogue, index)
    domain_dials = {}  # domain -> dialogue -> index -> turn_id
    for dialogue, tid, index, domain in zip(
        turns.dialogue, turns.turn_id, turns.turn_index, turns.domain
    ):
        if domain not in domain_dials:
            domain_dials[domain] = {}
        if dialogue not in domain_dials[domain]:
            domain_dials[domain][dialogue] = {}
        domain_dials[domain][dialogue][index] = tid
        turn_location[tid] = (domain, dialogue, index)
    new = []
    examples = data.predictions.slot_value_id << data.slot_values.slot_value_id
    for tid, sid, slot, value in zip(
        examples.turn_id, examples.slot_id, examples.slot, examples.value
    ):
        if value != req_token:
            domain, dialogue, index = turn_location[tid]
            c_idx = index + 2
            while c_idx in domain_dials[domain][dialogue]:
                cid = domain_dials[domain][dialogue][c_idx]
                new.append(dict(slot=slot, value=value, turn_id=cid, slot_id=sid, slot_value_id=None))
                c_idx += 2
        else:
            pass
    if pcent_of_continuation is not None:
        num = int(len(new) * pcent_of_continuation)
        new = rng.sample(new, num)
    elif pcent_of_existing is not None:
        num = int(original_pred_len * pcent_of_existing)
        new = rng.sample(new, num)
    else:
        raise ValueError('Either pcent_of_continuation or pcent_of_existing must be set')
    slot_values = dst.SlotValue.of(new)
    predictions = dst.Prediction.of(dict(
        slot_value_id=slot_values.slot_value_id, slot_id=slot_values.slot_id
    ), fill=None)
    data.slot_values += slot_values
    data.predictions += predictions
    return data


def downsample_domains(data, k=1):
    data: dst.Data
    turns_by_domain = {}
    for domain, tid in zip(data.turns.domain, data.turns.turn_id):
        if domain not in turns_by_domain:
            turns_by_domain[domain] = []
        turns_by_domain[domain].append(tid)
    downsampled_domains = rng.sample(list(turns_by_domain), k=k)
    downsampled_turn_ids = [tid for domain in downsampled_domains for tid in turns_by_domain[domain]]
    downsampled_turns = ~data.turns[downsampled_turn_ids]
    downsampled_turn_ids = set(downsampled_turn_ids)
    downsampled_slot_values = ~data.slot_values[[x in downsampled_turn_ids for x in data.slot_values.turn_id]]
    data.turns = downsampled_turns
    data.slot_values = downsampled_slot_values
    if data.predictions:
        downsampled_predictions = ~data.predictions[[
            x in downsampled_slot_values.slot_value_id for x in data.predictions.slot_value_id]]
        data.predictions = downsampled_predictions
    return data


def partition_by_domain(data, n_splits=1):
    data: dst.Data
    turns_by_domain = {}
    for domain, tid in zip(data.turns.domain, data.turns.turn_id):
        if domain not in turns_by_domain:
            turns_by_domain[domain] = []
        turns_by_domain[domain].append(tid)
    all_domains = list(turns_by_domain)
    rng.shuffle(all_domains)
    domain_splits = list(ez.batch(all_domains, math.ceil(len(all_domains) / n_splits)))
    partitions = []
    for domain_split in domain_splits:
        domain_split = set(domain_split)
        turn_ids_split = [
            tid for domain in domain_split for tid in turns_by_domain[domain]]
        turns_split = ~data.turns[turn_ids_split]
        turn_ids_split = set(turn_ids_split)
        slot_values_split = ~data.slot_values[[
            x in turn_ids_split for x in data.slot_values.turn_id]]
        slots_split = ~data.slots[[x in domain_split for x in data.slots.domain]]
        slot_ids_split = set(slots_split.slot_id)
        svids_split = set(slot_values_split.slot_value_id)
        value_cands_split = ~data.value_candidates[[
            x in slot_ids_split for x in data.value_candidates.slot_id]]
        split = dst.Data()
        split.turns = turns_split
        split.slot_values = slot_values_split
        split.slots = slots_split
        split.value_candidates = value_cands_split
        if data.predictions:
            preds_split = ~data.predictions[[
                x in svids_split for x in data.predictions.slot_value_id]]
            split.predictions = preds_split
        partitions.append(split)
    return partitions


def replace_to_continuation(data, req_token=None):
    data: dst.Data
    turns = data.turns
    turn_location = {}  # turn_id -> (domain, dialogue, index)
    domain_dials = {}  # domain -> dialogue -> index -> turn_id
    for dialogue, tid, index, domain in zip(
        turns.dialogue, turns.turn_id, turns.turn_index, turns.domain
    ):
        if domain not in domain_dials:
            domain_dials[domain] = {}
        if dialogue not in domain_dials[domain]:
            domain_dials[domain][dialogue] = {}
        domain_dials[domain][dialogue][index] = tid
        turn_location[tid] = (domain, dialogue, index)
    new = []
    examples = data.predictions.slot_value_id << data.slot_values.slot_value_id
    for tid, sid, slot, value, svid in zip(
        examples.turn_id, examples.slot_id, examples.slot, examples.value, examples.slot_value_id
    ):
        if value == req_token:
            new.append(dict(
                slot=slot, value=value, turn_id=tid, slot_id=sid, slot_value_id=svid))
            continue
        domain, dialogue, index = turn_location[tid]
        continuation = [(tid, svid)]
        c_idx = index + 2
        while c_idx in domain_dials[domain][dialogue]:
            cid = domain_dials[domain][dialogue][c_idx]
            continuation.append((cid, None))
            c_idx += 2
        c_turn_id, c_svid = rng.choice(continuation)
        new.append(dict(
            slot=slot, value=value, turn_id=c_turn_id, slot_id=sid, slot_value_id=c_svid))
    slot_values = dst.SlotValue.of(new)
    predictions = dst.Prediction.of(dict(
        slot_value_id=slot_values.slot_value_id, slot_id=slot_values.slot_id
    ), fill=None)
    data.slot_values = slot_values
    data.predictions = predictions
    return data



