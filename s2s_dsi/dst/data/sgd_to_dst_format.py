import json
import os
import pathlib
import pickle
import pathlib as pl
import random
import copy
import ezpyz as ez

import dst.data.dst_data as dst


def sgd_to_dst(data_json, slot_alternatives, use_sgdx=True, duplicate_for_alternatives=False):
    domain_fix_slotmap = {
        (slot.domain, slot.name): dst.Slot(
            name=slot.name,
            domain=slot.domain.split('_', 1)[0],
            description=slot.description,
            values=slot.values,
            categorical=slot.categorical
        )
        for slots in slot_alternatives.values()
        for slot in slots
    }
    ds = []
    for dialogue in data_json:
        if not use_sgdx:
            slotmaps = [{
                slot_key: alternatives[0]
                for slot_key, alternatives in slot_alternatives.items()
            }]
        elif duplicate_for_alternatives:
            slotmaps = []
            for version in zip(*slot_alternatives.values()):
                slotmap = {
                    slot_key: slot_alternative
                    for slot_key, slot_alternative
                    in zip(slot_alternatives.keys(), version)
                }
                slotmaps.append(slotmap)
        else:
            slotmaps = [{
                slot_key: random.choice(alterantives)
                for slot_key, alterantives in slot_alternatives.items()
            }]
        for slotmap in slotmaps:
            ts = []
            for turn in dialogue['turns']:
                ss = {}
                frames = turn['frames']
                for frame in frames:
                    actions = frame['actions']
                    domain = frame['service']
                    for action in actions:
                        act = action['act']
                        slot_name = action['slot']
                        if slot_name:
                            values = list(action['values'])
                            slot_key = (domain, slot_name)
                            if slot_key in slotmap:
                                slot = slotmap[slot_key]
                                if act.startswith('REQ'):
                                    values.append('?')
                                ss[slot] = values
                ss = {
                    domain_fix_slotmap[(slot.domain, slot.name)]: values
                    for slot, values in ss.items()
                }
                if turn['speaker'] == "SYSTEM":
                    t = dst.Turn(
                        turn=turn['utterance'],
                        speaker="SYSTEM",
                        listener="USER",
                        slots=ss
                    )
                    ts.append(t)
                elif turn['speaker'] == "USER":
                    t = dst.Turn(
                        turn=turn['utterance'],
                        speaker="USER",
                        listener="SYSTEM",
                        slots=ss
                    )
                    ts.append(t)
            ds.append(dst.Dialogue(turns=ts))
    data = dst.DstData(dialogues=ds, ontology=list(domain_fix_slotmap.values()))
    return data

def map_sgdx_schemas():
    sgd_dir = pathlib.Path('data/sgd-data')
    sgdx_dir = sgd_dir / 'sgd_x'
    train_dev_test_slotmaps = []
    for folder in ('train', 'dev', 'test'):
        slotmap = {}
        train_dev_test_slotmaps.append(slotmap)
        sgd_schema = ez.File(sgd_dir/folder/'schema.json').load()
        for version in range(1, 6):
            sgdx_path = sgdx_dir/'data'/f"v{version}"/folder/'schema.json'
            sgdx_schema = ez.File(sgdx_path).load()
            for sgd_service, sgdx_service in zip(sgd_schema, sgdx_schema):
                sgd_service_name = sgd_service['service_name']
                sgdx_service_name = sgdx_service['service_name']
                assert sgdx_service_name.startswith(sgd_service_name)
                sgd_slots = sgd_service['slots']
                sgdx_slots = sgdx_service['slots']
                for sgd_slot, sgdx_slot in zip(sgd_slots, sgdx_slots):
                    sgd_slot_name = sgd_slot['name']
                    sgdx_slot_name = sgdx_slot['name']
                    sgd_slot_description = sgd_slot['description']
                    sgdx_slot_description = sgdx_slot['description']
                    sgd_slot_categorical = sgd_slot['is_categorical']
                    sgdx_slot_categorical = sgdx_slot['is_categorical']
                    sgd_slot_possible_values = sgd_slot['possible_values']
                    sgdx_slot_possible_values = sgdx_slot['possible_values']
                    sgd_slot_key = (sgd_service_name, sgd_slot_name)
                    if sgd_slot_key not in slotmap:
                        slot = dst.Slot(
                            name=sgd_slot_name,
                            domain=sgd_service_name,
                            description=sgd_slot_description,
                            values=sgd_slot_possible_values,
                            categorical=sgd_slot_categorical
                        )
                        slotmap[sgd_slot_key] = [slot]
                    alternative_slot = dst.Slot(
                        name=sgdx_slot_name,
                        domain=sgdx_service_name,
                        description=sgdx_slot_description,
                        values=sgdx_slot_possible_values,
                        categorical=sgdx_slot_categorical
                    )
                    slotmap[sgd_slot_key].append(alternative_slot)
    return train_dev_test_slotmaps


def filter_domains(data, out_domains):
    out_domains = set(out_domains)
    data = copy.deepcopy(data)
    ds = []
    dialogues = data.dialogues
    ontology = data.ontology
    for dialogue in dialogues:
        dialogue: dst.Dialogue
        for turn in dialogue.turns:
            turn_domains = {
                slot.domain for slot in turn.slots
            } if turn.slots else set()
            if turn_domains & out_domains:
                turn.slots = None
        if any(turn.slots for turn in dialogue.turns):
            ds.append(dialogue)
    result = dst.DstData(dialogues=ds, ontology=ontology)
    return result

def sample_domains(data:dst.DstData, n):
    domains = list(data.domains())
    random.shuffle(domains)
    domains = domains[:n]
    unsampled_domains = list(set(data.domains()) - set(domains))
    return domains, unsampled_domains

def filter_slotless_turns_from_examples(data:dst.DstData, kept_slotless_turns_per_dialogue=0):
    data = copy.deepcopy(data)
    slotless = [
        turn for dialogue in data.dialogues for turn in dialogue.turns
        if turn.slots == {}
    ]
    random.shuffle(slotless)
    slotless = slotless[len(data.dialogues)*kept_slotless_turns_per_dialogue:]
    for turn in slotless:
        turn.slots = None
    return data

def evenly_downsample_dialogues(data:dst.DstData, n):
    domain_counts = {
        domain: 0 for domain in data.domains()
    }
    dialogues = data.domains()
    downsampled = {}
    while len(downsampled) < n:
        dialogue = None
        dialogue_id = None
        while dialogue_id is None or dialogue_id in downsampled:
            domain_to_sample = min(domain_counts, key=domain_counts.get)
            dialogue = dialogues[domain_to_sample].pop()
            if not dialogues[domain_to_sample]:
                del dialogues[domain_to_sample]
                del domain_counts[domain_to_sample]
            dialogue_id = id(dialogue)
        downsampled[dialogue_id] = dialogue
        for domain in dialogue.domains():
            if domain in domain_counts:
                domain_counts[domain] += 1
    result = dst.DstData(dialogues=list(downsampled.values()), ontology=data.ontology)
    return result


def evenly_downsample_sgd_train():
    data = dst.DstData.load('data/sgd-data/sgdx_train_filtered.pkl')
    data_100 = evenly_downsample_dialogues(data, 100)
    data_100.save('data/sgd-data/sgdx_100_train.pkl')
    data_1k = evenly_downsample_dialogues(data, 1000)
    data_1k.save('data/sgd-data/sgdx_1k_train.pkl')


def main():
    train_slotmap, dev_slotmap, test_slotmap = map_sgdx_schemas()
    train_json = ez.File('data/sgd-data/train/train.json').load()
    dev_json = ez.File('data/sgd-data/dev/dev.json').load()
    test_json = ez.File('data/sgd-data/test/test.json').load()
    # train_json = random.sample(train_json, 100)
    # dev_json = random.sample(dev_json, 100)
    # test_json = random.sample(test_json, 100)

    dst_train_original = sgd_to_dst(train_json, train_slotmap, use_sgdx=False)
    dst_train = sgd_to_dst(train_json, train_slotmap)
    dst_train_6 = sgd_to_dst(train_json, train_slotmap, duplicate_for_alternatives=True)
    dst_valid = sgd_to_dst(dev_json, dev_slotmap, use_sgdx=False)
    dst_valid.save('data/sgd-data/sgd_valid.pkl')
    pure_sgd_test = sgd_to_dst(test_json, test_slotmap, use_sgdx=False)
    pure_sgd_test.save('data/sgd-data/sgd_test.pkl')
    dst_test = sgd_to_dst(test_json, test_slotmap)
    all_test_domains = set(dst_test.domains())
    all_train_domains = set(dst_train.domains())
    test_domains = list(all_test_domains - all_train_domains)
    train_domains = list(all_train_domains - set(test_domains))
    ez.File('data/sgd-data/sampled_domains.json').save(dict(train=train_domains, test=test_domains))
    dst_train_original = filter_domains(dst_train_original, test_domains)
    dst_train = filter_domains(dst_train, test_domains)
    dst_train_6 = filter_domains(dst_train_6, test_domains)
    dst_test = filter_domains(dst_test, train_domains)
    dst_train_original_filtered = filter_slotless_turns_from_examples(
        dst_train_original, kept_slotless_turns_per_dialogue=2
    )
    dst_train_filtered = filter_slotless_turns_from_examples(
        dst_train, kept_slotless_turns_per_dialogue=2
    )
    dst_train_6_filtered = filter_slotless_turns_from_examples(
        dst_train_6, kept_slotless_turns_per_dialogue=2
    )

    for data, path in [
        (dst_test, 'data/sgd-data/sgdx_test.pkl'),
        (dst_train_original, 'data/sgd-data/sgd_train.pkl'),
        (dst_train_original_filtered, 'data/sgd-data/sgd_train_filtered.pkl'),
        (dst_train, 'data/sgd-data/sgdx_train.pkl'),
        (dst_train_filtered, 'data/sgd-data/sgdx_train_filtered.pkl'),
        (dst_train_6, 'data/sgd-data/sgdx_6_train.pkl'),
        (dst_train_6_filtered, 'data/sgd-data/sgdx_6_train_filtered.pkl'),
    ]:
        data: dst.DstData
        print(path)
        excluded = [
            turn for dialogue in data.dialogues for turn in dialogue.turns
            if turn.slots is None
        ]
        empty = [
            turn for dialogue in data.dialogues for turn in dialogue.turns
            if turn.slots == {}
        ]
        filled = [
            turn for dialogue in data.dialogues for turn in dialogue.turns
            if turn.slots
        ]
        slots = [
            slot for dialogue in data.dialogues for turn in dialogue.turns
            if turn.slots is not None for slot in turn.slots
        ]
        slot_names = {
            slot.name for slot in slots
        }
        domains = set(data.domains())
        print(f'    Dialogues: {len(data.dialogues)}')
        print(f'    Domains: {", ".join(domains)}')
        print(f'    Excluded Turns: {len(excluded)}')
        print(f'    Empty Turns: {len(empty)}')
        print(f'    Turns with Examples: {len(filled) + len(empty)}')
        print(f'    Slots: {len(slots)}')
        print(f'    Unique Slot Names: {len(slot_names)}')
        data.save(path)



if __name__ == "__main__":
    main()