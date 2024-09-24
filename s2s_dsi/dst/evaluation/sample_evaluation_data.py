
from dst.approaches.dst_seq_data import DstSeqData
import ezpyz as ez
import random


def sample_evaluation_data(
    original_data_path,
    num_dialogues,
    new_data_path,
    even_domain_sampling=True,
    size_turn_window_sampling=None,
    skip_speaker=None,
    valid_data_too=False,
    seed=42,
):
    random.seed(seed)
    if isinstance(original_data_path, str):
        data = DstSeqData.load(original_data_path)
    else:
        data = original_data_path
    dialogues = list(data.dialogues)
    if size_turn_window_sampling is not None:
        if skip_speaker:
            size_turn_window_sampling *= 2
        for j, dialogue in reversed(list(enumerate(dialogues))):
            for i in range(len(dialogue.turns) - size_turn_window_sampling):
                window = dialogue.turns[i:i + size_turn_window_sampling + 1]
                if all(
                    turn.slots is not None or turn.speaker == skip_speaker
                    for turn in window
                ) and sum(int(bool(turn.slots)) for turn in window) >= 2:
                    break
            else: # nobreak
                del dialogues[j]
    if even_domain_sampling:
        domain_dialogues = {}
        for dialogue in dialogues:
            domains = dialogue.domains()
            domains = frozenset(domains)
            domain_dialogues.setdefault(domains, []).append(dialogue)
        single_domain_dialogues = {}
        for domains, dialogues in domain_dialogues.items():
            if len(domains) == 1:
                domain, = domains
                single_domain_dialogues[domain] = dialogues
        all_domains = set().union(*domain_dialogues)
        num_dials_per_domain = num_dialogues // len(all_domains)
        remainder = num_dialogues % len(all_domains)
        test_sample = []
        valid_sample = []
        domain_samples = {}
        for i, domain in enumerate(all_domains):
            num_dials = num_dials_per_domain + (1 if i < remainder else 0)
            domain_dials = single_domain_dialogues[domain]
            domain_sample = random.sample(domain_dials, min(num_dials*2, len(domain_dials)))
            domain_samples[domain] = domain_sample
            test_sample.extend(domain_sample[:num_dials])
            valid_sample.extend(domain_sample[num_dials:])
    elif valid_data_too:
        sample = random.sample(dialogues, num_dialogues*2)
        test_sample = sample[:num_dialogues]
        valid_sample = sample[num_dialogues:]
    else:
        test_sample = random.sample(dialogues, num_dialogues)
        valid_sample = []
    if valid_data_too:
        valid_sample = DstSeqData(valid_sample, ontology=data.ontology)
        valid_sample.save(str(new_data_path).replace('test', 'valid'))
    test_sample = DstSeqData(test_sample, ontology=data.ontology)
    test_sample.save(new_data_path)
    return test_sample

def combine_datasets(*dataset_paths):
    datasets = [DstSeqData.load(path) for path in dataset_paths]
    slots = [slot for dataset in datasets for slot in dataset.ontology.slots()]
    combined = DstSeqData(
        [dialogue for dataset in datasets for dialogue in dataset.dialogues],
        ontology=slots
    )
    return combined


if __name__ == '__main__':

    gptdst5k_test = combine_datasets(
        'data/gptdst5k/gptdst5k_test_domains_0.pkl',
        'data/gptdst5k/gptdst5k_valid_domains_0.pkl'
    )

    gptdst5k = sample_evaluation_data(
        gptdst5k_test, 100,
        'data/gptdst5k/gptdst5k_100_test.pkl',
        even_domain_sampling=True,
        size_turn_window_sampling=3,
        valid_data_too=True
    )
    mwoz = sample_evaluation_data(
        'data/mwz2.4/mwoz24_test.pkl', 100,
        'data/mwz2.4/mwoz24_100_test.pkl',
        even_domain_sampling=True,
        size_turn_window_sampling=3,
        skip_speaker='Clerk',
        valid_data_too=True
    )
    sgd = sample_evaluation_data(
        'data/sgd-data/sgdx_test.pkl', 100,
        'data/sgd-data/sgd_100_test.pkl',
        even_domain_sampling=True,
        size_turn_window_sampling=3,
        valid_data_too=True
    )
