
import ezpyz as ez
import pathlib as pl
from dst.approaches.dst_seq_data import DstSeqData


def filter_dg5k_domains(data_path, excluded_domains_path):
    excluded = set(list(zip(*ez.File(excluded_domains_path).load()[1:]))[0])
    data = DstSeqData.load(data_path)
    data.dialogues = [
        dialogue for dialogue in data.dialogues
        if dialogue.domains() and not any(domain in excluded for domain in dialogue.domains())
    ]
    data.save(data_path.replace('.pkl', '_filtered.pkl'))



if __name__ == '__main__':
    filter_dg5k_domains('data/gptdst5k/gptdst5k.pkl', 'data/excluded_domains.csv')