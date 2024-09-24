

from dst.approaches.dst_seq_data import DstSeqData
from bert_score import score as bert_score

import logging
import transformers
transformers.tokenization_utils.logger.setLevel(logging.ERROR)
transformers.configuration_utils.logger.setLevel(logging.ERROR)
transformers.modeling_utils.logger.setLevel(logging.ERROR)







if __name__ == '__main__':
    toy = DstSeqData([

    ])


    def test_bert_score():
        p, r, f1 = bert_score(cands=[
                'Chicken: yes',
                'time of arrival: 6:00',
                'number of choices: 8',
                'number of choices: 8',
                'number of choices: 8',
                'number of choices: 8',
            ],
            refs=[
                'Reason for visit: to speak about the chicken',
                'arrival time: 6:00',
                'option count: 8',
                'number of options: eight',
                'number of chickens: several',
                'choices: 8'
            ],
            lang='en',
            verbose=False,
            rescale_with_baseline=True,
        )
        print(f1)
