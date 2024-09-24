

import dataclasses as dc
import ezpyzy as ez
import dsi.cluster as dcl
import difflib as dl
import torch as pt
from dsi.bert_encoder import BertValueEncoder
from fuzzywuzzy.fuzz import partial_ratio as fuzz_partial_ratio, ratio as fuzz_ratio
from fuzzywuzzy.process import extractOne as fuzz_extract
from collections import Counter
import bisect as bs

excluded_domains = []


@dc.dataclass
class SimMatcher:
    cosine_matcher_threshold: float = 0.8
    pred_cluster_slot_values_match_to_gold_threshold: float = 80.0
    cluster_precision: float|None = None
    cluster_recall: float|None = None
    cluster_f1: float|None = None
    n: int|None = None
    best_matches: dict = None
    value_precision: float|None = None
    value_recall: float|None = None
    value_f1: float|None = None
    pred_svids_with_no_match: list[str] = None

    def __post_init__(self):
        self.bert = BertValueEncoder()
        self.best_matches = {}

    def match_values(self, gold: dcl.ClusteredPoint, pred: dcl.ClusteredPoint):
        gold = ~gold[[all(d not in s for d in excluded_domains) for s in gold.slot]]
        gold_names = {c: n for c, n in zip(gold.cluster_id, gold.slot)}
        pred_names = {c: n for c, n in zip(pred.cluster_id, pred.slot)}
        gold_value_encodings = self.bert.encode(values=gold.value)
        pred_value_encodings = self.bert.encode(values=pred.value)
        gold_by_cluster = {}
        for cluster_id, encoding in zip(gold.cluster_id, gold_value_encodings):
            if cluster_id not in gold_by_cluster:
                gold_by_cluster[cluster_id] = []
            gold_by_cluster[cluster_id].append(encoding)
        gold_centroids = {}
        for gold_cluster, gold_encodings in gold_by_cluster.items():
            centroid = pt.mean(pt.stack(gold_encodings), dim=0)
            gold_centroids[gold_cluster] = centroid
        pred_by_cluster = {}
        for cluster_id, encoding in zip(pred.cluster_id, pred_value_encodings):
            if cluster_id not in pred_by_cluster:
                pred_by_cluster[cluster_id] = []
            pred_by_cluster[cluster_id].append(encoding)
        pred_centroids = {}
        for pred_cluster, pred_encodings in pred_by_cluster.items():
            centroid = pt.mean(pt.stack(pred_encodings), dim=0)
            pred_centroids[pred_cluster] = centroid
        for pred_cluster, pred_centroid in pred_centroids.items():
            if pred_cluster == -1:
                continue
            best_match = None
            best_similarity = self.cosine_matcher_threshold
            for gold_cluster, gold_centroid in gold_centroids.items():
                similarity = pt.cosine_similarity(pred_centroid, gold_centroid, dim=0)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = gold_cluster
            self.best_matches[pred_cluster] = best_match
        self.cluster_precision = len({p for p,g in self.best_matches.items() if g is not None}) /\
                                 len(set(pred.cluster_id))
        self.cluster_recall = len({g for p,g in self.best_matches.items() if g is not None}) /\
                                 len(set(gold.cluster_id))
        self.cluster_f1 = 2 * self.cluster_precision * self.cluster_recall /\
                            (self.cluster_precision + self.cluster_recall) if self.cluster_precision and self.cluster_recall else 0.0
        gold_values_by_cluster = {}
        for cluster_id, value in zip(gold.cluster_id, gold.value):
            if cluster_id not in gold_values_by_cluster:
                gold_values_by_cluster[cluster_id] = []
            gold_values_by_cluster[cluster_id].append(value)
        pred_values_by_cluster = {}
        for cluster_id, value in zip(pred.cluster_id, pred.value):
            if cluster_id not in pred_values_by_cluster:
                pred_values_by_cluster[cluster_id] = []
            pred_values_by_cluster[cluster_id].append(value)

        value_precisions = {}   # gold slot -> precision for that slot
        value_recalls = {}      # gold slot -> recall for that slot
        value_f1s = []
        gold_to_preds = ez.reverse_map(self.best_matches)
        gold_to_preds.pop(None, None)
        for gold_cluster, pred_clusters in gold_to_preds.items():
            big_ass_pred_cluster = [pv.lower() for pc in pred_clusters for pv in pred_values_by_cluster[pc]]
            gold_overlap = Counter()
            pred_overlap = Counter()
            gold_value_counts = Counter([x.lower() for x in gold_values_by_cluster[gold_cluster] if x != 'any'])
            pred_value_counts = Counter(big_ass_pred_cluster)
            for pred_value, pred_value_count in pred_value_counts.items():
                best_gold_value_match, best_gold_value_count, best_match_score = None, 0, -1
                for gold_value, gold_value_count in gold_value_counts.items():
                    fuzz_score = fuzz_partial_ratio(gold_value, pred_value)
                    if (fuzz_score >= self.pred_cluster_slot_values_match_to_gold_threshold and
                        fuzz_score > best_match_score or (
                            fuzz_score == best_match_score and gold_value_count > best_gold_value_count
                        )):
                        best_gold_value_match = gold_value
                        best_gold_value_count = gold_value_count
                        best_match_score = fuzz_score
                # we have the best gold match for the pred value, or None if nothing matched
                if best_gold_value_match is not None:
                    gold_overlap[best_gold_value_match] += best_gold_value_count
                    pred_overlap[pred_value] += pred_value_count
            # calculate the precision and recall for this gold slot
            value_precision_denom = 0
            value_precision_num = 0
            for pred_value, pred_value_count in pred_value_counts.items():
                value_precision_denom += pred_value_count
                value_precision_num += pred_overlap[pred_value]
            value_precisions[gold_cluster] = value_precision_num / value_precision_denom if value_precision_denom else 0.0
            value_recall_denom = 0
            value_recall_num = 0
            for gold_value, gold_value_count in gold_value_counts.items():
                value_recall_denom += gold_value_count
                value_recall_num += min(gold_overlap[gold_value], gold_value_count)
            value_recalls[gold_cluster] = value_recall_num / value_recall_denom if value_recall_denom else 0.0
            value_f1 = 2 * value_precisions[gold_cluster] * value_recalls[gold_cluster] /\
                        (value_precisions[gold_cluster] + value_recalls[gold_cluster]) if\
                        value_precisions[gold_cluster] and value_recalls[gold_cluster] else 0.0
            value_f1s.append(value_f1)
        self.value_precision = sum(value_precisions.values()) / len(value_precisions) if value_precisions else 0.0
        self.value_recall = sum(value_recalls.values()) / len(value_recalls) if value_recalls else 0.0
        self.value_f1 = sum(value_f1s) / len(value_f1s) if value_f1s else 0.0

        #
        # precisions = []
        # recalls = []
        # for pred_cluster, gold_cluster in self.best_matches.items():
        #     if gold_cluster is None:
        #         continue
        #     gold_matched = set()
        #     pred_matched = set()
        #     gold_values = gold_values_by_cluster[gold_cluster]
        #     pred_values = pred_values_by_cluster[pred_cluster]
        #     for pred_val in pred_values:
        #         # matches = dl.get_close_matches(pred_val.lower(), gold_values, cutoff=self.pred_cluster_slot_values_match_to_gold_threshold)
        #         matches = {}
        #         for gold_value in gold_values:
        #             ratio = fuzz.partial_ratio(gold_value, pred_val)
        #             if ratio >= self.pred_cluster_slot_values_match_to_gold_threshold:
        #                 matches[gold_value] = ratio
        #         if matches:
        #             # gold_match = matches[0]
        #             gold_match = max(matches, key=matches.get)
        #             gold_matched.add(gold_match)
        #             pred_matched.add(pred_val)
        #     precision = len(pred_matched) / len(set(pred_values))
        #     recall = len(gold_matched) / len(set(gold_values))
        #     precisions.append(precision)
        #     recalls.append(recall)
        # self.value_precision = sum(precisions) / len(precisions) if precisions else 0.0
        # self.value_recall = sum(recalls) / len(recalls) if recalls else 0.0
        # self.value_f1 = 2 * self.value_precision * self.value_recall / (self.value_precision + self.value_recall) if self.value_precision and self.value_recall else 0.0
        self.n = len(set(pred.cluster_id) - {None, -1})
        return self.best_matches

