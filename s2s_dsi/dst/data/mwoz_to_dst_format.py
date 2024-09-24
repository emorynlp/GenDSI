import copy

import dst.data.dst_data as dst
import json
from dst.data.split import leave_one_out_splits, duplicate_dialogues_per_dialogue_domain

def split(s, *splitters):
    parts = [s]
    for splitter in splitters:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(splitter))
        parts = new_parts
    return parts

def postprocess_slot_value(value: str|list[str]) -> list[str]|None:
    if value is None:
        return None
    elif isinstance(value, list):
        values = []
        for v in value:
            values.extend(postprocess_slot_value(v))
        return values
    else:
        parts = split(value, '|', '<', '>')
        parts = [', '.join(parts)]
        values = []
        for part in parts:
            part = part.strip()
            if part:
                values.append(part)
        return values

def mwoz_to_dst(dialogues: list, ontology: dict) -> dst.DstData:
    ds = []
    ts = []
    o = {}
    slots_per_domain = {'police': set()}
    for domain_slotname, candidates in ontology.items():
        domain_slotname = domain_slotname.lower()
        slot_domain, slot_name = domain_slotname.split('-')
        s = dst.Slot(
            name=domain_slotname,
            domain=slot_domain,
            description=my_mwoz_ontology[slot_name.lower()].format(domain=slot_domain),
            values=postprocess_slot_value(candidates),
            categorical=True
        )
        o[domain_slotname] = s
        slots_per_domain.setdefault(slot_domain, set()).add(s)
    for dialogue in dialogues:
        i = 0
        for turn in dialogue['dialogue']:
            system_turn = turn['system_transcript']
            belief_state = turn['belief_state']
            slots = turn['turn_label']
            user_turn = turn['transcript']
            domain = turn['domain']
            ss = {}
            for name, value in slots:
                values = postprocess_slot_value(value)
                s = o[name]
                # if set(values) - set(s.values):
                #     s.values.extend(value for value in values if value not in s.values)
                ss[s] = values
            ss = {**dict.fromkeys(slots_per_domain[domain]), **ss} # add negatives
            if system_turn:
                st = dst.Turn(
                    turn=system_turn,
                    speaker='Clerk',
                    listener='Tourist'
                )
                i += 1
                ts.append(st)
            ut = dst.Turn(
                turn=user_turn,
                speaker='Tourist',
                listener='Clerk',
                slots=ss
            )
            ts.append(ut)
            i += 1
        ds.append(dst.Dialogue(ts))
        ts = []
    slot_list = []
    for slot_name, slot in o.items():
        slot_list.append(slot)
    data = dst.DstData(dialogues=ds, ontology=slot_list)
    return data


my_mwoz_ontology = {
    'name': 'The name of the {domain}',
    'internet': 'Whether the {domain} has internet access',
    'type': 'The type of {domain}',
    'departure': "The location the {domain} will depart",
    'destination': "The location the {domain} will arrive at",
    'pricerange': "The price range category of the {domain}",
    'area': "The general area the {domain} is in",
    'book people': "The number of people the {domain} booking is for",
    'arriveby': "The time the {domain} should arrive at the destination",
    'food': "The ethnicity of food served at the {domain}",
    'book stay': "The number of nights the {domain} booking is for",
    'book day': "The day of the week of the {domain} booking",
    'day': "The day of the week of the {domain} booking",
    'leaveat': "The time the {domain} should leave the departure location",
    'department': "The department of the {domain}",
    'stars': "The star rating of the {domain}",
    'parking': "Whether the {domain} has parking",
    'book time': "The time of the {domain} booking",
}


def load_mwoz(path):
    with open(path) as f:
        return json.load(f)


if __name__ == '__main__':

    ontology = load_mwoz('data/mwz2.4/ontology.json')

    mwoz_train = load_mwoz('data/mwz2.4/train_dials.json')
    mwoz_dev = load_mwoz('data/mwz2.4/dev_dials.json')
    mwoz_test = load_mwoz('data/mwz2.4/test_dials.json')
    mwoz_full = mwoz_train + mwoz_dev + mwoz_test

    dst_valid = mwoz_to_dst(mwoz_dev, ontology)
    dst_train = mwoz_to_dst(mwoz_train, ontology)
    dst_test = mwoz_to_dst(mwoz_test, ontology)
    dst_full = mwoz_to_dst(mwoz_full, ontology)

    leave_one_out_by_domain = leave_one_out_splits(dst_full)

    dst_valid = duplicate_dialogues_per_dialogue_domain(dst_valid)
    dst_train = duplicate_dialogues_per_dialogue_domain(dst_train)
    dst_test = duplicate_dialogues_per_dialogue_domain(dst_test)
    dst_full = duplicate_dialogues_per_dialogue_domain(dst_full)

    dst_train.save('data/mwz2.4/mwoz24_train.pkl')
    dst_valid.save('data/mwz2.4/mwoz24_dev.pkl')
    dst_test.save('data/mwz2.4/mwoz24_test.pkl')
    dst_full.save('data/mwz2.4/mwoz24_full.pkl')

    for domain, (train, test) in leave_one_out_by_domain.items():
        train.save(f'data/mwz2.4/mwoz24_train_{domain}.pkl')
        test.save(f'data/mwz2.4/mwoz24_test_{domain}.pkl')







