"""
Loading completeness auto metrics...
[bertscore]
	threshold=0.6271260380744934 achieves f1=0.599
	acl1: 0.73795 (0.69269,0.78321) p=0.0
	kalpha: 0.19799 (0.10771,0.28827) p=1.894253297241022e-05
[gpt]
	acl1: 0.68785 (0.64211,0.7336) p=0.0
	kalpha: 0.25516 (0.17785,0.33246) p=1.5311996115485726e-10
[rouge]
	threshold=0.2857142984867096 achieves f1=0.610
	acl1: 0.68159 (0.63048,0.7327) p=0.0
	kalpha: 0.21993 (0.1343,0.30556) p=5.803637719292709e-07
[sacrebleu]
	threshold=15.090767860412598 achieves f1=0.585
	acl1: 0.66558 (0.61318,0.71799) p=0.0
	kalpha: 0.17002 (0.08566,0.25438) p=8.339990554051546e-05
[sbert]
	threshold=0.44981956481933594 achieves f1=0.637
	acl1: 0.76806 (0.72576,0.81037) p=0.0
	kalpha: 0.27501 (0.1817,0.36832) p=1.067986055147685e-08
      sacrebleu rouge bertscore sbert   gpt
Alpha      0.17  0.22      0.20  0.28  0.26
AC1        0.67  0.68      0.74  0.77  0.69

                      sacrebleu_gslots  rouge_gslots  bertscore_gslots  sbert_gslots    gpt
dataname modelname
gpt      GPTpipeline               NaN           NaN               NaN           NaN    NaN
         gpt                       NaN           NaN               NaN           NaN    NaN
         sgd                       NaN           NaN               NaN           NaN    NaN
sgd      GPTpipeline            27.200         0.487             0.714         0.636  0.898
         gpt                    29.363         0.502             0.720         0.642  0.890
         sgd                    33.311         0.476             0.764         0.627  0.848


Calculating correctness auto metrics...
Running auto metric (rouge): 100%|████████████████| 1/1 [00:01<00:00,  1.00s/it]
Running auto metric (sacrebleu): 100%|████████████| 1/1 [00:10<00:00, 10.80s/it]
Running auto metric (bertscore): 100%|████████████| 1/1 [00:16<00:00, 16.18s/it]
Running auto metric (sbert): 100%|████████████████| 1/1 [00:03<00:00,  3.37s/it]
Running auto metric (rouge): 100%|████████████████| 1/1 [00:01<00:00,  1.47s/it]
Running auto metric (sacrebleu): 100%|████████████| 1/1 [00:18<00:00, 18.60s/it]
Running auto metric (bertscore): 100%|████████████| 1/1 [00:15<00:00, 15.38s/it]
Running auto metric (sbert): 100%|████████████████| 1/1 [00:05<00:00,  5.44s/it]
Running auto metric (rouge): 100%|████████████████| 1/1 [00:01<00:00,  1.36s/it]
Running auto metric (sacrebleu): 100%|████████████| 1/1 [00:16<00:00, 16.51s/it]
Running auto metric (bertscore): 100%|████████████| 1/1 [00:13<00:00, 13.72s/it]
Running auto metric (sbert): 100%|████████████████| 1/1 [00:05<00:00,  5.10s/it]
[bertscore]
	threshold=0.4193984568119049 achieves f1=0.649
	acl1: 0.76049 (0.73396,0.78701) p=0.0
	kalpha: 0.29723 (0.24102,0.35345) p=0.0
[gpt]
	acl1: 0.39331 (0.34965,0.43697) p=0.0
	kalpha: 0.16953 (0.12363,0.21543) p=6.257216966787382e-13
[rouge]
	threshold=0.0 achieves f1=0.631
	acl1: 0.61117 (0.5758,0.64653) p=0.0
	kalpha: 0.26253 (0.21318,0.31188) p=0.0
[sacrebleu]
	threshold=7.809849739074707 achieves f1=0.629
	acl1: 0.62433 (0.58966,0.65899) p=0.0
	kalpha: 0.25744 (0.20757,0.30731) p=0.0
[sbert]
	threshold=0.1883271187543869 achieves f1=0.663
	acl1: 0.74738 (0.71988,0.77489) p=0.0
	kalpha: 0.32576 (0.27118,0.38033) p=0.0
      sacrebleu rouge bertscore sbert   gpt
Alpha      0.26  0.26      0.30  0.33  0.17
AC1        0.62  0.61      0.76  0.75  0.39

                      sacrebleu  rouge  bertscore  sbert    gpt
dataname modelname
gpt      GPTpipeline        NaN    NaN        NaN    NaN    NaN
         gpt                NaN    NaN        NaN    NaN    NaN
         sgd                NaN    NaN        NaN    NaN    NaN
sgd      GPTpipeline      5.146  0.092      0.159  0.136  0.236
         gpt              5.709  0.099      0.157  0.139  0.231
         sgd             12.306  0.177      0.268  0.226  0.868

Process finished with exit code 0

"""

import sys
sys.path.append('/local/scratch/sfillwo/diverse-state-tracking-refactor/src')

import pandas as pd
from dst.approaches.dst_seq_data import DstSeqData
import ezpyz as ez
from dst.evaluation.data_to_annotation_json import modelmap
from dst.evaluation.auto_eval_baseline import AutoEvalBaseline, BleuMetric, SacreBleuMetric, RougeMetric, BertScoreMetric, SbertMetric
from sklearn.metrics import accuracy_score, f1_score
import torch
import pprint as pp
import irrCAC.raw as cac
from dst.evaluation.gpt_auto_eval import predicted_slot_matches_reference

print(f"Cuda: {torch.cuda.is_available()}")

def to_binary(full_score_df: pd.DataFrame, label: str):
    """
    score_df: DataFrame of scores
    label: column of ground truth label
    """
    to_convert = sorted(set(full_score_df.columns) - {label})
    latex = {}
    for metric in to_convert:
        if 'gslots' not in metric:
            print(f'[{metric}]')
            if metric != 'gpt':
                score_df = full_score_df[[label, metric]].dropna()
                metric_scores = score_df[metric]
                labels = score_df[label]
                ascending_metrics = sorted(set(metric_scores))
                threshold, best_perf = None, 0
                for candidate in ascending_metrics:
                    predictions = (metric_scores > candidate).astype(int)
                    perf = f1_score(labels, predictions, average='macro')
                    if perf > best_perf:
                        best_perf = perf
                        threshold = candidate
                print(f'\tthreshold={threshold} achieves f1={best_perf:.3f}')
                predictions = (metric_scores > threshold).astype(int)
                score_df[f'{metric}_binary'] = predictions
                df = score_df[[label, f'{metric}_binary']]
            else:
                score_df = full_score_df[[label, metric]].dropna()
                df = score_df[[label, metric]]
            agreement_obj = cac.CAC(df)
            gwet = agreement_obj.gwet()['est']
            print(f"\tacl1: {gwet['coefficient_value']} ({gwet['confidence_interval'][0]},{gwet['confidence_interval'][1]}) p={gwet['p_value']}")
            kalpha = agreement_obj.krippendorff()['est']
            print(f"\tkalpha: {kalpha['coefficient_value']} ({kalpha['confidence_interval'][0]},{kalpha['confidence_interval'][1]}) p={kalpha['p_value']}")
            latex[metric.replace('_gslots', '')] = {
                'Alpha': f"{kalpha['coefficient_value']:.2f}",
                'AC1': f"{gwet['coefficient_value']:.2f}"
            }
    print(pd.DataFrame.from_dict(latex)[['sacrebleu', 'rouge', 'bertscore', 'sbert', 'gpt']])

def get_model_data(key):
    if key in modelmap:
        return pd.Series((modelmap[key][0], modelmap[key][1][:3]), index=['modelname', 'dataname'])
    else:
        return pd.Series([None, None], index=['modelname', 'dataname'])

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

def model_metrics(full_score_df: pd.DataFrame):
    df = full_score_df.reset_index()
    # break apart index col into columns: modelkey, didx, tidx, sidx (if exists)
    split_df = df['index'].str.split('-', expand=True)
    split_df.columns = ['modelkey', 'didx', 'tidx', 'sidx']
    df = pd.concat([df, split_df], axis=1)
    # add modelname and dataname columns based on modelkey
    df[['modelname', 'dataname']] = df['modelkey'].apply(get_model_data)
    # groupby([dataname, modelname])
    filtered_df = df.drop(columns=['index', 'modelkey', 'didx', 'tidx', 'sidx'])
    filtered_df['modelname'] = filtered_df['modelname'].replace({'DaringWookiee': 'gpt', 'FearlessBodhi': 'sgd'})
    grouped_df = filtered_df.groupby(['dataname', 'modelname'])
    # micro average over metric columns
    grouped_avgs = grouped_df.aggregate('mean')
    grouped_avgs = grouped_avgs.round(3)
    
    columns = {x for x in grouped_avgs.columns if 'gslots' in x}
    columns = columns if columns else set(grouped_avgs.columns) - {'correct', 'complete'}
    columns.add('gpt')
    
    print()
    order = ['sacrebleu', 'rouge', 'bertscore', 'sbert', 'gpt']
    print(grouped_avgs[[c for o in order for c in columns if o in c]])

RUN_GPT = False
RUN_AUTO_COMPLETE = True
RUN_AUTO_CORRECT = True
TESTDATA = 'sgd'

if __name__ == '__main__':
    print()

    ###############
    # GPT
    ###############

    if RUN_GPT:

        if ez.File(f'data/dsg_predictions/scores/completeness_from_gpt_{TESTDATA}.pkl').path.exists():
            print('Loading gpt as eval...')
            completeness_with_gpt_df = ez.File(f'data/dsg_predictions/scores/completeness_from_gpt_{TESTDATA}.pkl').load()
            correctness_with_gpt_df = ez.File(f'data/dsg_predictions/scores/correctness_from_gpt_{TESTDATA}.pkl').load()
        else:
            print('Calculating gpt as eval...')
            completeness_scores = ez.File('data/dsg_predictions/scores/completeness.pkl').load()
            completeness_with_gpt_df = pd.DataFrame()
            correctness_scores = ez.File('data/dsg_predictions/scores/correctness.pkl').load()
            correctness_with_gpt_df = pd.DataFrame()
            for modelkey, (modelname, dataname) in modelmap.items():
                if TESTDATA in dataname:
                    model_completeness_df = pd.DataFrame()
                    model_correctness_df = pd.DataFrame()
                    data = DstSeqData.load(f"data/dsg_predictions/{modelname}/{dataname}")
                    eval_with_metric = AutoEvalBaseline(predicted_slot_matches_reference)
                    model_correctness_gpt_df, model_completeness_gpt_df = eval_with_metric.gpt_as_eval(data, modelkey)
                    completeness_with_gpt_df = pd.concat([completeness_with_gpt_df, model_completeness_gpt_df], axis=0)
                    correctness_with_gpt_df = pd.concat([correctness_with_gpt_df, model_correctness_gpt_df], axis=0)
            completeness_with_gpt_df = pd.concat([completeness_with_gpt_df, pd.DataFrame({'complete': completeness_scores})], axis=1)
            correctness_with_gpt_df = pd.concat([correctness_with_gpt_df, pd.DataFrame({'correct': correctness_scores})], axis=1)
            ez.File(f'data/dsg_predictions/scores/correctness_from_gpt_{TESTDATA}.pkl').save(correctness_with_gpt_df)
            ez.File(f'data/dsg_predictions/scores/completeness_from_gpt_{TESTDATA}.pkl').save(completeness_with_gpt_df)


    ###############
    # Completeness
    ###############

    if RUN_AUTO_COMPLETE:

        if ez.File(f'data/dsg_predictions/scores/completeness_with_auto_{TESTDATA}.pkl').path.exists():
            print('Loading completeness auto metrics...')
            completeness_with_auto_df = ez.File(f'data/dsg_predictions/scores/completeness_with_auto_{TESTDATA}.pkl').load()
        else:
            print('Calculating completeness auto metrics...')
            completeness_scores = ez.File('data/dsg_predictions/scores/completeness.pkl').load()
            completeness_with_auto_df = pd.DataFrame()

            for modelkey, (modelname, dataname) in modelmap.items():
                if TESTDATA in dataname:
                    data = DstSeqData.load(f"data/dsg_predictions/{modelname}/{dataname}")
                    df = pd.DataFrame()
                    for metric in [
                        RougeMetric,
                        SacreBleuMetric,
                        BertScoreMetric,
                        SbertMetric
                    ]:
                        eval_with_metric = AutoEvalBaseline(metric())
                        result = eval_with_metric.eval_wrt_ref(data, modelkey)
                        df = pd.concat([df, result], axis=1)
                    completeness_with_auto_df = pd.concat([completeness_with_auto_df, df], axis=0)
            completeness_with_auto_df = pd.concat([completeness_with_auto_df, pd.DataFrame({'complete': completeness_scores})], axis=1)
            ez.File(f'data/dsg_predictions/scores/completeness_with_auto_{TESTDATA}.pkl').save(completeness_with_auto_df)

        if RUN_GPT:
            completeness_with_auto_df = pd.concat([completeness_with_auto_df, completeness_with_gpt_df['gpt']], axis=1)
        to_binary(completeness_with_auto_df, label='complete')
        model_metrics(completeness_with_auto_df)
        print()
        print()

    ###############
    # Correctness
    ###############

    if RUN_AUTO_CORRECT:

        if ez.File(f'data/dsg_predictions/scores/correctness_with_auto_{TESTDATA}.pkl').path.exists():
            print('Loading correctness auto metrics...')
            correctness_with_auto_df = ez.File(f'data/dsg_predictions/scores/correctness_with_auto_{TESTDATA}.pkl').load()
        else:
            print('Calculating correctness auto metrics...')
            correctness_scores = ez.File('data/dsg_predictions/scores/correctness.pkl').load()
            correctness_with_auto_df = pd.DataFrame()

            for modelkey, (modelname, dataname) in modelmap.items():
                if TESTDATA in dataname:
                    data = DstSeqData.load(f"data/dsg_predictions/{modelname}/{dataname}")
                    df = pd.DataFrame()
                    for metric in [
                        RougeMetric,
                        SacreBleuMetric,
                        BertScoreMetric,
                        SbertMetric
                    ]:
                        eval_with_metric = AutoEvalBaseline(metric())
                        result = eval_with_metric.eval_wrt_pred(data, modelkey)
                        df = pd.concat([df, result], axis=1)
                    correctness_with_auto_df = pd.concat([correctness_with_auto_df, df], axis=0)
            correctness_with_auto_df = pd.concat([correctness_with_auto_df, pd.DataFrame({'correct': correctness_scores})], axis=1)
            ez.File(f'data/dsg_predictions/scores/correctness_with_auto_{TESTDATA}.pkl').save(correctness_with_auto_df)

        if RUN_GPT:
            correctness_with_auto_df = pd.concat([correctness_with_auto_df, correctness_with_gpt_df['gpt']], axis=1)
        to_binary(correctness_with_auto_df, label='correct')
        model_metrics(correctness_with_auto_df)
