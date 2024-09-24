import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from dst.approaches.dst_seq_data import DstSeqData

import evaluate

from rouge_score.rouge_scorer import RougeScorer

import bert_score
from transformers import BertTokenizer, BertModel

from sentence_transformers import SentenceTransformer, util


class BleuMetric:
    """
    """

    def __init__(self) -> None:
        self.name = 'bleu'
        self.model = evaluate.load('bleu')

    def __call__(self, refs: list[str], preds: list[str]):
        """
        bleu score matrix of ref x pred
        """
        assert isinstance(refs, list) and isinstance(preds, list)
        scores = [
            self.model.compute(predictions=[pred.lower()], references=[ref.lower()], max_order=1)['bleu'] 
            for pred, ref in zip(preds, refs)
        ]
        scores_tensor = torch.tensor(scores)
        return scores_tensor

class SacreBleuMetric:
    """
    Sacrebleu
        - calculated using 1-grams to 4-grams
        - If multiple references provided, then the score is the MAX, not average

    Documentation references:
    https://huggingface.co/docs/datasets/how_to_metrics
    https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/bleu.py
    https://github.com/mjpost/sacreBLEU -> use None or '' to fill in reference_lists to be same length if doing batch processing
    """

    def __init__(self) -> None:
        self.name = 'sacrebleu'
        self.model = evaluate.load('sacrebleu')

    def __call__(self, refs, preds):
        """
        sacrebleu score matrix of ref x pred
        """
        assert isinstance(refs, list) and isinstance(preds, list)
        scores = [
            self.model.compute(predictions=[pred.lower()], references=[ref.lower()], lowercase=True)['score'] 
            for pred, ref in zip(preds, refs)
        ]
        scores_tensor = torch.tensor(scores)
        return scores_tensor


class RougeMetric:

    def __init__(self, n="L") -> None:
        """
        n: type of Rouge (1 vs 2 vs L)
        """
        self.model = RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.n = n
        self.name = 'rouge'

    def __call__(self, refs, preds): 
        """
        rouge score matrix of ref x pred
        """
        assert isinstance(refs, list) and isinstance(preds, list)
        scores = [
            self.model.score(ref.lower(), pred.lower())[f"rouge{self.n}"].fmeasure
            for pred, ref in zip(preds, refs)
        ]
        scores_tensor = torch.tensor(scores)
        return scores_tensor


class BertScoreMetric:
    """
    picked rank1 bertscore model according to:
        https://github.com/Tiiiger/bert_score 
        https://docs.google.com/spreadsheets/d/1RKOVpselB98Nnh_EOC4A2BYn8_201tmPODpNWu4w7xI/edit#gid=0
    """

    def __init__(self) -> None:
        self.name = 'bertscore'

    def __call__(self, refs, preds): 
        """
        bertscore score matrix of ref x pred
        """
        p, r, f1 = bert_score.score(
            preds, 
            refs, 
            model_type='microsoft/deberta-xlarge-mnli', 
            lang='en'
        )
        return f1


class SbertMetric:
    """
    picked general purpose rank1 model from https://www.sbert.net/docs/pretrained_models.html
    """

    def __init__(self) -> None:
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.name = 'sbert'

    def __call__(self, refs, preds): 
        """
        sbert score matrix of ref x pred
        """
        refs_embed = self.model.encode(refs, convert_to_tensor=True)
        preds_embed = self.model.encode(preds, convert_to_tensor=True)
        cosine_sim = util.pairwise_cos_sim(refs_embed, preds_embed)
        return cosine_sim



class AutoEvalBaseline:
    def __init__(self, metric, bs=40960):
        self.metric: callable = metric
        self.bs = bs

    def eval_wrt_ref(self, data: DstSeqData, modelkey: str): 
        # todo - micro average for recall metric on model-level

        """
        Calculate metrics where the gold labels are the focus,
        e.g. find maximum predicted label match for each gold label to measure completeness

        Returns:
            pandas dataframe: dialogue turn x avg metric score
        """
 
        turn_metric_dict = {}
        slot_metric_dict = {}

        # flatten slot comparisons
        ref_slots_flat = []
        pred_slots_flat = []
        for didx, dialogue in enumerate(data.dialogues):
            for tidx, turn in enumerate(dialogue.turns): 
                refs = [
                        slot.name + ": " + ", ".join(slot_value) for slot, slot_value in turn.slots.items() if slot_value is not None
                    ] if turn.slots is not None else []
                preds = [
                        slot.name + ": " + ", ".join(slot_value) for slot, slot_value in turn.predicted_slots.items() if slot_value is not None
                    ] if turn.predicted_slots is not None else []
                if refs and preds:
                    preds_ls = [pred for ref in refs for pred in preds]
                    refs_ls = [ref for ref in refs for pred in preds]
                    ref_slots_flat.extend(refs_ls)
                    pred_slots_flat.extend(preds_ls)
        assert len(ref_slots_flat) == len(pred_slots_flat)

        # batch and pass to metric
        scores_flat = []
        for j in tqdm(range(0, len(ref_slots_flat), self.bs), desc=f'Running auto metric ({self.metric.name})'):
            ref_batch = ref_slots_flat[j: min(j+self.bs, len(ref_slots_flat))]
            pred_batch = pred_slots_flat[j: min(j+self.bs, len(pred_slots_flat))]
            scores = self.metric(ref_batch, pred_batch)
            scores_flat.extend(scores.tolist())

        # reconstruct scores into slots comparisons per turn per dialogue
        curr_flat_idx = 0
        for didx, dialogue in enumerate(data.dialogues):
            for tidx, turn in enumerate(dialogue.turns): 
                refs = [
                        slot.name + ": " + ", ".join(slot_value) for slot, slot_value in turn.slots.items() if slot_value is not None
                    ] if turn.slots is not None else []
                preds = [
                        slot.name + ": " + ", ".join(slot_value) for slot, slot_value in turn.predicted_slots.items() if slot_value is not None
                    ] if turn.predicted_slots is not None else []
                if refs and preds:
                    preds_ls = [pred for ref in refs for pred in preds]
                    scores = torch.tensor(scores_flat[curr_flat_idx:curr_flat_idx+len(preds_ls)]).view(len(refs), len(preds))
                     # max of each row
                    max_values, max_indices = torch.max(scores, dim=1)
                    turn_metric = torch.mean(max_values).item()
                    for sidx, slot_metric in enumerate(max_values):
                        slot_metric_dict[f"{modelkey}-{didx}-{tidx}-{sidx}"] = slot_metric.item()
                    curr_flat_idx += len(preds_ls)
                elif refs:
                    turn_metric = 0
                    for sidx in range(len(refs)):
                        slot_metric_dict[f"{modelkey}-{didx}-{tidx}-{sidx}"] = 0
                else:
                    continue
                turn_metric_dict[f"{modelkey}-{didx}-{tidx}"] = turn_metric
        assert curr_flat_idx == len(scores_flat)

        metric_df = pd.DataFrame({
            self.metric.name: pd.Series(turn_metric_dict), 
            f'{self.metric.name}_gslots': slot_metric_dict
        })
        return metric_df

    def eval_wrt_pred(self, data: DstSeqData, modelkey: str): 
        """
        Calculate metrics where the predicted labels are the focus,
        e.g. find maximum gold label match for each predicted label to measure correctness

        Returns:
            pandas dataframe: dialogue turn slot x metric score
        """
 
        metric_dict = {}

        # flatten slot comparisons
        ref_slots_flat = []
        pred_slots_flat = []
        for didx, dialogue in enumerate(data.dialogues):
            for tidx, turn in enumerate(dialogue.turns): 
                refs = [
                        slot.name + ": " + ", ".join(slot_value) for slot, slot_value in turn.slots.items() if slot_value is not None
                    ] if turn.slots is not None else []
                preds = [
                        slot.name + ": " + ", ".join(slot_value) for slot, slot_value in turn.predicted_slots.items() if slot_value is not None
                    ] if turn.predicted_slots is not None else []
                if refs and preds:
                    preds_ls = [pred for pred in preds for ref in refs]
                    refs_ls = [ref for pred in preds for ref in refs]
                    ref_slots_flat.extend(refs_ls)
                    pred_slots_flat.extend(preds_ls)
        assert len(ref_slots_flat) == len(pred_slots_flat)

        # batch and pass to metric
        scores_flat = []
        for j in tqdm(range(0, len(ref_slots_flat), self.bs), desc=f'Running auto metric ({self.metric.name})'):
            ref_batch = ref_slots_flat[j: min(j+self.bs, len(ref_slots_flat))]
            pred_batch = pred_slots_flat[j: min(j+self.bs, len(pred_slots_flat))]
            scores = self.metric(ref_batch, pred_batch)
            scores_flat.extend(scores.tolist())

        # reconstruct scores into slots comparisons
        curr_flat_idx = 0
        for didx, dialogue in enumerate(data.dialogues):
            for tidx, turn in enumerate(dialogue.turns): 
                refs = [
                        slot.name + ": " + ", ".join(slot_value) for slot, slot_value in turn.slots.items() if slot_value is not None
                    ] if turn.slots is not None else []
                preds = [
                        slot.name + ": " + ", ".join(slot_value) for slot, slot_value in turn.predicted_slots.items() if slot_value is not None
                    ] if turn.predicted_slots is not None else []
                if refs and preds:
                    preds_ls = [pred for pred in preds for ref in refs]
                    scores = torch.tensor(scores_flat[curr_flat_idx:curr_flat_idx+len(preds_ls)]).view(len(preds), len(refs))
                     # max of each row
                    max_values, max_indices = torch.max(scores, dim=1)
                    for sidx, slot_metric in enumerate(max_values):
                        metric_dict[f"{modelkey}-{didx}-{tidx}-{sidx}"] = slot_metric.item()
                    curr_flat_idx += len(preds_ls)
                elif preds:
                    for sidx in range(len(preds)):
                        metric_dict[f"{modelkey}-{didx}-{tidx}-{sidx}"] = 0
                else:
                    continue
        assert curr_flat_idx == len(scores_flat)

        metric_df = pd.DataFrame({self.metric.name: pd.Series(metric_dict)})
        return metric_df
    
    def gpt_as_eval(self, data: DstSeqData, modelkey: str): 
        correctness_dict = {}
        completeness_dict = {}
        # missed = {}
        for didx, dialogue in tqdm(list(enumerate(data.dialogues)), desc='running gpt as eval'):
            for tidx, turn in enumerate(dialogue.turns): 
                if turn.slots is not None:
                    # call the prompt function
                    ref_has_match, pred_has_match = self.metric(turn, log=print, debug=True)
                    # correctness
                    for sidx, (slot, value) in enumerate(turn.predicted_slots.items()):
                        if value is not None:
                            correctness_dict[f"{modelkey}-{didx}-{tidx}-{sidx}"] = int((turn, slot) in pred_has_match)
                    # completeness
                    # complete -> all slots in turn.slots are in matches
                    is_complete = True
                    for sidx, (slot, value) in enumerate(turn.slots.items()):
                        if value is not None:
                            completeness_dict[f"{modelkey}-{didx}-{tidx}-{sidx}"] = int((turn, slot) in ref_has_match)
                            if (turn, slot) not in ref_has_match:
                                is_complete = False
                                # missed.setdefault(','.join(value), 0)
                                # missed[','.join(value)] += 1
                    completeness_dict[f"{modelkey}-{didx}-{tidx}"] = int(is_complete)
        correctness_df = pd.DataFrame({'gpt': pd.Series(correctness_dict)})
        completeness_df = pd.DataFrame({'gpt': pd.Series(completeness_dict)})
        return correctness_df, completeness_df