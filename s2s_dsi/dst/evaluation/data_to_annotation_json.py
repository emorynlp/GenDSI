import textwrap
import irrCAC.raw as cac
import ezpyz as ez
from dst.data.dst_data import DstData
import dst.approaches.dst_seq_data as dstseq
import pandas as pd
import pprint as pp
import itertools as it
import numpy as np
import random
import statsmodels.stats.proportion as sm
from scipy import stats
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

def propci_wilson_cc(count, nobs, alpha=0.05):
    # get confidence limits for proportion
    # using wilson score method w/ cont correction
    # i.e. Method 4 in Newcombe [1];
    n = nobs
    p = count/n
    q = 1.-p
    z = stats.norm.isf(alpha / 2.)
    z2 = z**2
    denom = 2*(n+z2)
    num = 2.*n*p+z2-1.-z*np.sqrt(z2-2-1./n+4*p*(n*q+1))
    ci_l = num/denom
    num = 2.*n*p+z2+1.+z*np.sqrt(z2+2-1./n+4*p*(n*q-1))
    ci_u = num/denom
    if p == 0:
        ci_l = 0.
    elif p == 1:
        ci_u = 1.
    return ci_l, ci_u


modelmap = {
    'A': ('FearlessBodhi', 'gptdst5k_100_test.dstseqdata'),
    'B': ('GPTpipeline', 'gptdst5k_100_test.dstseqdata'),
    'C': ('DaringWookiee', 'gptdst5k_100_test.dstseqdata'),
    'D': ('FearlessBodhi', 'sgd_100_test.dstseqdata'),
    'E': ('GPTpipeline', 'sgd_100_test.dstseqdata'),
    'F': ('DaringWookiee', 'sgd_100_test.dstseqdata'),
    # 'G': ('DaringWookiee', 'gptdst5k_100_valid.dstseqdata')
}



def data_to_annoation_json(turn_window=None, seed=42):
    random.seed(seed)
    collection = []
    for modelid, (experiment, dataname) in modelmap.items():
        prediction_path = f"ex/Seq2seqDSG/{experiment}/1/predictions/{dataname}"
        predictions = DstData.load(prediction_path)
        for i, dialogue in enumerate(predictions.dialogues):
            if turn_window is not None:
                turn_windows = []
                for j in range(len(dialogue.turns) - turn_window):
                    window = dialogue.turns[j:j + turn_window]
                    if all(
                        turn.slots is not None
                            for turn in window
                    ) and sum(int(bool(turn.slots)) for turn in window) >= 2:
                        turn_windows.append(tuple(range(j, j + turn_window)))
                selected_window = random.choice(turn_windows)
            else:
                selected_window = tuple(range(len(dialogue.turns)))
            turns = []
            for j, turn in enumerate(dialogue.turns):
                slots = None
                if turn.predicted_slots is not None:
                    slots = []
                    for slot, values in turn.predicted_slots.items():
                        if values is not None:
                            slotvalue = dict(
                                name = slot.name,
                                value = ', '.join(values),
                            )
                            slots.append(slotvalue)
                turn = dict(
                    text = turn.turn,
                    slots = slots,
                    skip = j not in selected_window
                )
                turns.append(turn)
            dialogue = dict(
                turns = turns,
                model = modelid,
                index = i,
            )
            collection.append(dialogue)
    random.seed(seed)
    random.shuffle(collection)
    data = dict(
        collection = collection,
    )
    return data


def split_data_per_annotator(data, *annotator_names, overlap=10):
    dialogues: list = data['collection']
    overlapping = dialogues[:overlap]
    remaining = dialogues[overlap:]
    per_annotator = {name: list(overlapping) for name in annotator_names}
    num_per_annotator = len(remaining) // len(annotator_names)
    if len(remaining) % len(annotator_names):
        num_per_annotator += 1
    for i, (annotator, tasks) in enumerate(per_annotator.items()):
        start = i * num_per_annotator
        end = start + num_per_annotator
        tasks.extend(remaining[start:end])
    for name, tasks in per_annotator.items():
        task_data = dict(collection = tasks)
        ez.File(f'src/annotation/annotation/round1_{name}.json').save(task_data)

def load_data_per_annotator(
    path_prefix,
    *annotator_names,
    data_prefix='round1'
):
    per_annotator = {}
    correctness_annotations = {}
    completeness_annotations = {}
    for name in annotator_names:
        print(f"{name:=^80}")
        annotations = ez.File(f'{path_prefix}/{data_prefix}/{data_prefix}_{name}.json').load()['collection']
        per_annotator[name] = annotations
        for dialogue in annotations:
            model, index = dialogue['model'], dialogue['index']
            turns = dialogue['turns']
            for t, turn in enumerate(turns):
                text = turn['text']
                slots = turn['slots']
                is_complete = turn['state_is_complete']
                if is_complete is not None:
                    print('✅ ' if is_complete else '❌ ', "\n".join(textwrap.wrap(text, 80, subsequent_indent='   ')))
                    completeness_annotations.setdefault(f'{model}-{index}-{t}', {})[name] = is_complete
                else:
                    print('|> ', "\n".join(textwrap.wrap(text, 80, subsequent_indent='   ')))
                if any(
                    slot['is_correct'] is not None for slot in slots
                ):
                    for s, slot in enumerate(slots):
                        item_id = f'{model}-{index}-{t}-{s}'
                        print(f"        {slot['name']}: {slot['value']} "
                              f"{'✅' if slot['is_correct'] else '❌'}")
                        correctness_annotations.setdefault(item_id, {})[name] = slot['is_correct']
    # df: dataframe -> item x rater-decision
    for annotation_name, annotation_records in [
        ('completeness', completeness_annotations),
        ('correctness', correctness_annotations)
    ]:
        print(f"{annotation_name:=^80}")
        df = pd.DataFrame.from_records([
            dict(item=key, **value) for key, value in annotation_records.items()
        ])
        count_overlapping_items = df.notna().sum(axis=1).gt(3).sum().item()
        print(f"Got {count_overlapping_items} items for {annotation_name} annotations")
        df.set_index('item', inplace=True)
        df = df.astype(float)
        agreement_obj = cac.CAC(df)
        gwet = agreement_obj.gwet()
        pp.pp(gwet)
        kalpha = agreement_obj.krippendorff()
        pp.pp(kalpha)
        value_counts_per_column = df.apply(lambda col: col.value_counts(), axis=0)
        print('\n\n', 'Value counts')
        print(value_counts_per_column)
        df['approach'] = [f"{modelmap[x][0]}({modelmap[x][1].split('.')[0]})" for x in df.index.str[:1]]
        scores = df[[*annotator_names]].apply(lambda row: row.mode().max(), axis=1)
        '''Scores!!!'''
        ez.File(f"data/dsg_predictions/scores/{annotation_name}.pkl").save(scores)
        turns_with_issues = {}
        for model_key, (model_name, data_name) in modelmap.items():
            data = dstseq.DstSeqData.load(f"data/dsg_predictions/{model_name}/{data_name}")
            for item, score in zip(scores.index, scores):
                if not score:
                    modelkey, dindex, tindex, *_ = item.split('-')
                    if model_name == modelmap.get(modelkey)[0] and data_name == modelmap.get(modelkey)[1]:
                        dialogue = data.dialogues[int(dindex)]
                        turn = dialogue.turns[int(tindex)]
                        turns_with_issues.setdefault((model_name, data_name), set()).add(turn)
        for (model_name, data_name), errored_turns in turns_with_issues.items():
            data_save_name = data_name.split('.')[0]
            error_analysis_data = random.sample(list(errored_turns), min(100, len(errored_turns)))
            save_path = f"data/dsg_predictions/error_analysis/{model_name}/{data_save_name}.pkl"
            ez.File(save_path).save(error_analysis_data)
        print('\n\n', f'Proportion of good turns for {annotation_name}')
        nobs = scores.groupby(df['approach']).count()
        successes = scores.groupby(df['approach']).sum()
        nobs_and_successes = pd.DataFrame(dict(nobs=nobs, successes=successes))
        def get_ci(row):
            pos = row['successes']
            n = row['nobs']
            point = row['successes'] / row['nobs']
            ci = sm.proportion_confint(row['successes'], row['nobs'], alpha=0.05, method='agresti_coull')
            return pd.Series([point, *ci, n, pos], index=['prop', 'ci_low', 'ci_high', 'n', 'pos'])
        point_hi_low = nobs_and_successes.apply(get_ci, axis=1)
        print(point_hi_low)
        print('\n\n')
        pairwise_p_values = pd.DataFrame(columns=['Approach 1', 'Approach 2', 'p-value'])
        for pair in it.combinations(point_hi_low.index, 2):
            approach1, approach2 = pair
            idx1 = point_hi_low.index == approach1
            idx2 = point_hi_low.index == approach2
            n1 = point_hi_low.loc[idx1, 'n'].values[0]
            n2 = point_hi_low.loc[idx2, 'n'].values[0]
            pos1 = point_hi_low.loc[idx1, 'pos'].values[0]
            pos2 = point_hi_low.loc[idx2, 'pos'].values[0]
            z_stat, p_value = sm.test_proportions_2indep(
                nobs1=n1, nobs2=n2,
                count1=pos1, count2=pos2,
                method='agresti-caffo', return_results=False
            )
            pair_result = pd.DataFrame({'Approach 1': [approach1], 'Approach 2': [approach2], 'p-value': [p_value]})
            pairwise_p_values = pd.concat([pairwise_p_values, pair_result])
        print(pairwise_p_values)



if __name__ == '__main__':
    # data = data_to_annoation_json(turn_window=3)
    # data['collection'] = data['collection']
    # split_data_per_annotator(data,
    #     'peace', 'michelle', 'helen',
    #     overlap=20
    # )
    # ...
    load_data_per_annotator(
        'src/annotation',
        'peace',
        'helen',
        'michelle',
        data_prefix='round1'
    )

    # evaluation dialogues from SGD test: 100
    # evaluation dialogues from GPTDST test: 104

'''
==================================completeness==================================
Got 60 items for completeness annotations
{'est': {'coefficient_value': 0.63465,
         'coefficient_name': 'AC1',
         'confidence_interval': (0.43432, 0.83497),
         'p_value': 6.418441333977398e-10,
         'z': 6.21361,
         'se': 0.10214,
         'pa': 0.75556,
         'pe': 0.33094},
 'weights': array([[1., 0.],
       [0., 1.]]),
 'categories': [0.0, 1.0]}
{'est': {'coefficient_value': 0.2702,
         'coefficient_name': "Krippendorff's Alpha",
         'confidence_interval': (0.06366, 0.47674),
         'p_value': 0.011227496384453861,
         'z': 2.61778,
         'se': 0.10322,
         'pa': 0.75691,
         'pe': 0.66691},
 'weights': array([[1., 0.],
       [0., 1.]]),
 'categories': [0.0, 1.0]}


 Value counts
     peace  helen  michelle
1.0    521    488       509
0.0    121    148       133


 Proportion of good turns for completeness
                                      prop    ci_low   ci_high      n    pos
approach                                                                    
DaringWookiee(gptdst5k_100_test)  0.956667  0.926596  0.975191  300.0  287.0
DaringWookiee(sgd_100_test)       0.946667  0.914530  0.967509  300.0  284.0
FearlessBodhi(gptdst5k_100_test)  0.323333  0.272879  0.378255  300.0   97.0
FearlessBodhi(sgd_100_test)       0.693333  0.638927  0.742851  300.0  208.0
GPTpipeline(gptdst5k_100_test)    0.933333  0.898763  0.956946  300.0  280.0
GPTpipeline(sgd_100_test)         0.900000  0.860465  0.929420  300.0  270.0



                         Approach 1                        Approach 2        p-value
0  DaringWookiee(gptdst5k_100_test)       DaringWookiee(sgd_100_test)   5.800310e-01
0  DaringWookiee(gptdst5k_100_test)  FearlessBodhi(gptdst5k_100_test)  1.071896e-100
0  DaringWookiee(gptdst5k_100_test)       FearlessBodhi(sgd_100_test)   3.201377e-19
0  DaringWookiee(gptdst5k_100_test)    GPTpipeline(gptdst5k_100_test)   2.222504e-01
0  DaringWookiee(gptdst5k_100_test)         GPTpipeline(sgd_100_test)   8.061024e-03
0       DaringWookiee(sgd_100_test)  FearlessBodhi(gptdst5k_100_test)   1.803424e-94
0       DaringWookiee(sgd_100_test)       FearlessBodhi(sgd_100_test)   2.336992e-17
0       DaringWookiee(sgd_100_test)    GPTpipeline(gptdst5k_100_test)   5.024985e-01
0       DaringWookiee(sgd_100_test)         GPTpipeline(sgd_100_test)   3.452186e-02
0  FearlessBodhi(gptdst5k_100_test)       FearlessBodhi(sgd_100_test)   2.616376e-22
0  FearlessBodhi(gptdst5k_100_test)    GPTpipeline(gptdst5k_100_test)   6.142304e-87
0  FearlessBodhi(gptdst5k_100_test)         GPTpipeline(sgd_100_test)   3.334434e-71
0       FearlessBodhi(sgd_100_test)    GPTpipeline(gptdst5k_100_test)   3.826236e-15
0       FearlessBodhi(sgd_100_test)         GPTpipeline(sgd_100_test)   1.063242e-10
0    GPTpipeline(gptdst5k_100_test)         GPTpipeline(sgd_100_test)   1.461871e-01
==================================correctness===================================
Got 140 items for correctness annotations
{'est': {'coefficient_value': 0.73155,
         'coefficient_name': 'AC1',
         'confidence_interval': (0.59291, 0.87018),
         'p_value': 0.0,
         'z': 10.34535,
         'se': 0.07071,
         'pa': 0.80952,
         'pe': 0.29046},
 'weights': array([[1., 0.],
       [0., 1.]]),
 'categories': [0.0, 1.0]}
{'est': {'coefficient_value': 0.43107,
         'coefficient_name': "Krippendorff's Alpha",
         'confidence_interval': (0.2967, 0.56545),
         'p_value': 2.962171841147665e-09,
         'z': 6.34261,
         'se': 0.06796,
         'pa': 0.80998,
         'pe': 0.666},
 'weights': array([[1., 0.],
       [0., 1.]]),
 'categories': [0.0, 1.0]}


 Value counts
     peace  helen  michelle
1.0   1268   1265      1115
0.0    293    141       359


 Proportion of good turns for correctness
                                      prop    ci_low   ci_high      n    pos
approach                                                                    
DaringWookiee(gptdst5k_100_test)  0.811912  0.785893  0.835437  957.0  777.0
DaringWookiee(sgd_100_test)       0.817043  0.788677  0.842371  798.0  652.0
FearlessBodhi(gptdst5k_100_test)  0.725926  0.669738  0.775775  270.0  196.0
FearlessBodhi(sgd_100_test)       0.907895  0.874316  0.933309  380.0  345.0
GPTpipeline(gptdst5k_100_test)    0.819473  0.794208  0.842258  986.0  808.0
GPTpipeline(sgd_100_test)         0.846753  0.819535  0.870528  770.0  652.0



                         Approach 1                        Approach 2       p-value
0  DaringWookiee(gptdst5k_100_test)       DaringWookiee(sgd_100_test)  7.889083e-01
0  DaringWookiee(gptdst5k_100_test)  FearlessBodhi(gptdst5k_100_test)  3.616343e-03
0  DaringWookiee(gptdst5k_100_test)       FearlessBodhi(sgd_100_test)  1.380087e-06
0  DaringWookiee(gptdst5k_100_test)    GPTpipeline(gptdst5k_100_test)  6.673809e-01
0  DaringWookiee(gptdst5k_100_test)         GPTpipeline(sgd_100_test)  5.632940e-02
0       DaringWookiee(sgd_100_test)  FearlessBodhi(gptdst5k_100_test)  2.446522e-03
0       DaringWookiee(sgd_100_test)       FearlessBodhi(sgd_100_test)  1.007759e-05
0       DaringWookiee(sgd_100_test)    GPTpipeline(gptdst5k_100_test)  8.885123e-01
0       DaringWookiee(sgd_100_test)         GPTpipeline(sgd_100_test)  1.168265e-01
0  FearlessBodhi(gptdst5k_100_test)       FearlessBodhi(sgd_100_test)  4.495519e-09
0  FearlessBodhi(gptdst5k_100_test)    GPTpipeline(gptdst5k_100_test)  1.473842e-03
0  FearlessBodhi(gptdst5k_100_test)         GPTpipeline(sgd_100_test)  5.209146e-05
0       FearlessBodhi(sgd_100_test)    GPTpipeline(gptdst5k_100_test)  6.873106e-06
0       FearlessBodhi(sgd_100_test)         GPTpipeline(sgd_100_test)  2.492198e-03
0    GPTpipeline(gptdst5k_100_test)         GPTpipeline(sgd_100_test)  1.302210e-01
'''

'''
Error analysis of DaringWookiee contained 355 slots
Error analysis of FearlessBodhi contained 107 slots
Error analysis of GPTpipeline contained 398 slots
'''