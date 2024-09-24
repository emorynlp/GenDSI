
import dst.data.dst_data as dst
import random


def remove_out_of_domain_slots(dialogue:dst.Dialogue, domain:str):
    for turn in (t for t in dialogue.turns if t.slots is not None):
        if any(
            slot.domain != domain
            for slot in turn.slots # noqa
        ):
            turn.slots = None

def duplicate_dialogues_per_dialogue_domain(data: dst.DstData):
    domain_aware = dst.DstData([], ontology=data.ontology)
    for dialogue in data.dialogues:
        domains = set(
            slot.domain
            for turn in dialogue.turns
            if turn.slots is not None
            for slot in turn.slots
        )
        for domain in domains:
            new_dialogue = domain_aware.add(dialogue)
            remove_out_of_domain_slots(new_dialogue, domain)
    return domain_aware

def leave_one_out_splits(data: dst.DstData):
    domains = data.ontology.domains()
    splits: dict[str, tuple[dst.DstData,dst.DstData]] = {
        domain: (
            dst.DstData([], ontology=data.ontology),
            dst.DstData([], ontology=data.ontology)
        )
        for domain in domains
    }
    for domain, domain_slots in domains.items():
        for dialogue in data.dialogues:
            dial_domains = set(
                slot.domain
                for turn in dialogue.turns
                if turn.slots is not None
                for slot in turn.slots
            )
            for dial_domain in dial_domains:
                if dial_domain != domain:
                    train_dialogue = splits[domain][0].add(dialogue)
                    remove_out_of_domain_slots(train_dialogue, dial_domain)
                else:
                    test_dialogue = splits[domain][1].add(dialogue)
                    remove_out_of_domain_slots(test_dialogue, domain)
    return splits

def leave_n_out_splits(data: dst.DstData, num_splits=10, valid_size=0, test_size=1, seed=None):
    random.seed(seed)
    splits = {}
    by_domain = data.domains()
    domains = list(by_domain.keys())
    random.shuffle(domains)
    test_domains = [domains[i*test_size:(i+1)*test_size] for i in range(num_splits)]
    for i, test_domain in enumerate(test_domains):
        train_domains = [domain for domain in domains if domain not in test_domain]
        random.shuffle(train_domains)
        valid_domains, train_domains = train_domains[:valid_size], train_domains[valid_size:]
        train = [dial for dom in train_domains for dial in by_domain[dom]]
        valid = [dial for dom in valid_domains for dial in by_domain[dom]]
        test = [dial for dom in test_domain for dial in by_domain[dom]]
        splits[i] = (
            dst.DstData(train, ontology=data.ontology),
            dst.DstData(valid, ontology=data.ontology),
            dst.DstData(test, ontology=data.ontology)
        )
    return splits


def random_split(data: dst.DstData, train_size=0.8, valid_size=0.1, test_size=0.1, seed=None):
    random.seed(seed)
    dialogues = data.dialogues
    random.shuffle(dialogues)
    if all(x <= 1 for x in (train_size, valid_size, test_size)):
        train = dialogues[:int(train_size * len(dialogues))]
        valid = dialogues[
            int(train_size * len(dialogues)):int((train_size + valid_size) * len(dialogues))
        ]
        test = dialogues[int((train_size + valid_size) * len(dialogues)):]
    else:
        train = dialogues[:train_size]
        valid = dialogues[train_size:train_size+valid_size]
        test = dialogues[train_size+valid_size:]
    train = dst.DstData(train, ontology=data.ontology)
    valid = dst.DstData(valid, ontology=data.ontology)
    test = dst.DstData(test, ontology=data.ontology)
    return train, valid, test
