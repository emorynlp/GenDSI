
import dataclasses as dc
import ezpyzy as ez
import dsi.cluster as dcl
import difflib as dl

excluded_domains = []

# by_domain_map = {
#     'attraction area': 'area',
#     'hotel area': 'area',
#     'restaurant area': 'area',
#     'hotel book day': 'book day',
#     'hotel book people': 'book people',
#     'hotel book stay': 'book stay duration',
#     'restaurant book day': 'book day',
#     'restaurant book people': 'book people',
#     'restaurant book time': 'reservation time',
#     'hotel price range': 'price range',
#     'restaurant price range': 'price range',
#     'taxi arrive by': 'arrive by',
#     'taxi departure': 'departure',
#     'taxi destination': 'destination',
#     'taxi leave at': 'leave at',
#     'train arrive by': 'arrive by',
#     'train book people': 'book people',
#     'train day': 'book day',
#     'train departure': 'departure',
#     'train destination': 'destination',
#     'train leave at': 'leave at',
#     'hotel stars': 'stars',
#     'restaurant stars': 'stars',
# }

@dc.dataclass
class Matcher:
    fuzzy_matcher_threshold: float = 0.5
    pred_cluster_slot_values_match_to_gold_threshold: float = 0.2
    cluster_precision: float|None = None
    cluster_recall: float|None = None
    cluster_f1: float|None = None
    n: int|None = None
    best_matches: dict = None
    value_precision: float|None = None
    value_recall: float|None = None
    value_f1: float|None = None
    pred_svids_with_no_match: list[str] = None

    def match(self,
        gold: dcl.ClusteredPoint,
        pred: dcl.ClusteredPoint,
        by_domain=True
    ):
        slot_to_domain = {s: d for s, d in zip(gold.slot, gold.domain) if d not in excluded_domains}

        pred_points = {} # turn -> slot -> (value, cluster)
        pred_clusters = {} # cluster -> (turn, slot, value)
        for cluster, slot, value, turn, svid in zip(
            pred.cluster_id, pred.slot, pred.value, pred.turn_id, pred.slot_value_id
        ):
            if cluster != -1:
                pred_points.setdefault(turn, {})[slot] = (value, cluster, svid)
                pred_clusters[cluster] = (turn, slot, value)
        gold_points = {}  # turn -> slot(domain) -> value
        for domain, slot, value, turn in zip(gold.domain, gold.slot, gold.value, gold.turn_id):
            if domain not in excluded_domains:
                gold_points.setdefault(turn, {})[slot] = value

        # for each turn
            # for each predicted slot-value, fuzzy match to gold slot-values of the turn
        fuzzy_matches = [] # pred sv string, gold sv string
        point_matches = [] # turn, pred_cluster, pred_svid, pred_slot, pred_value, gold_slot, gold_value
        no_match = [] # turn, pred_cluster, pred_svid, pred_slot, pred_value
        for turn, predicted_slot_values in pred_points.items():
            gold_slot_values = list(gold_points.get(turn, {}).items())
            gold_slot_value_strings = {f"{v}": i for i, (s, v) in enumerate(gold_slot_values)}
            for pred_slot, (pred_value, pred_cluster, pred_svid) in predicted_slot_values.items():
                matches = dl.get_close_matches(
                    f"{pred_value}", gold_slot_value_strings, cutoff=self.fuzzy_matcher_threshold)
                if matches:
                    gold_slot_value_string = matches[0]
                    gold_slot, gold_value = gold_slot_values[gold_slot_value_strings[gold_slot_value_string]]
                    fuzzy_matches.append((f"{pred_slot}: {pred_value}", gold_slot_value_string))
                    point_matches.append((turn, pred_cluster, pred_svid, pred_slot, pred_value, gold_slot, gold_value))
                else:
                    no_match.append(pred_svid)

        # assign each pred cluster to a gold slot based on number of matches
        pred_to_gold_counts = {} # pred cluster -> gold slot -> num matches
        for turn, pred_cluster, pred_svid, pred_slot, pred_value, gold_slot, gold_value in point_matches:
            if not by_domain:
                gold_slot = gold_slot.replace(slot_to_domain[gold_slot], '')
            pred_to_gold_counts.setdefault(pred_cluster, {}).setdefault(gold_slot, 0)
            pred_to_gold_counts[pred_cluster][gold_slot] += 1
        pred_cluster_to_gold_slot_matches = {} # pred cluster -> gold slot
        for pred_cluster, gold_slot_counts in pred_to_gold_counts.items():
            pred_cluster_to_gold_slot_matches[pred_cluster] = max(gold_slot_counts, key=gold_slot_counts.get)
            total_count = len([p for p in pred.cluster_id if p == pred_cluster])
            precentage_matching = gold_slot_counts[pred_cluster_to_gold_slot_matches[pred_cluster]] / total_count
            if precentage_matching < self.pred_cluster_slot_values_match_to_gold_threshold:
                del pred_cluster_to_gold_slot_matches[pred_cluster]

        # calculate slot p/r/f1
        gold_slots = {
            s if by_domain else s.replace(slot_to_domain[s], '')
            for s, d in zip(gold.slot, gold.domain) if d not in excluded_domains
        }
        matched_gold = set(pred_cluster_to_gold_slot_matches.values()) - {None}
        if len(pred_clusters) == 0:
            slot_precision = 0.0
        else:
            slot_precision = len(pred_cluster_to_gold_slot_matches) / len(pred_clusters)
        slot_recall = len(matched_gold) / len(gold_slots)
        print(f"Missed gold slots: {gold_slots - matched_gold}")
        print("\n".join(f"{k}: {v}" for k,v in pred_cluster_to_gold_slot_matches.items()))
        if slot_precision or slot_recall:
            slot_f1 = 2 * slot_precision * slot_recall / (slot_precision + slot_recall)
        else:
            slot_f1 = 0.0

        # calculate value p/r/f1 (matched pred clusters only)
        gold_value_was_matched_count = 0
        gold_value_total_count = 0
        pred_value_was_matched_count = 0
        pred_value_total_count = sum(
            len([x for x in pred.cluster_id if x == pc]) for pc in pred_cluster_to_gold_slot_matches)
        gold_slot_to_pred_clusters = {}
        for pred_cluster, gold_slot in pred_cluster_to_gold_slot_matches.items():
            gold_slot_to_pred_clusters.setdefault(gold_slot, set()).add(pred_cluster)
        for gold_cluster, pred_clusters in gold_slot_to_pred_clusters.items():
            for turn, pred_cluster, pred_svid, pred_slot, pred_value, gold_slot, gold_value in point_matches:
                if not by_domain:
                    gold_slot = gold_slot.replace(slot_to_domain[gold_slot], '')
                if gold_slot == gold_cluster:
                    gold_value_total_count += 1
                    if pred_cluster in pred_clusters:
                        gold_value_was_matched_count += 1
                        pred_value_was_matched_count += 1
        if pred_value_total_count:
            value_precision = pred_value_was_matched_count / pred_value_total_count
        else:
            value_precision = 0.0
        if gold_value_total_count:
            value_recall = gold_value_was_matched_count / gold_value_total_count
        else:
            value_recall = 0.0
        if value_precision or value_recall:
            value_f1 = 2 * value_precision * value_recall / (value_precision + value_recall)
        else:
            value_f1 = 0.0

        self.cluster_recall = slot_recall
        self.cluster_precision = slot_precision
        self.cluster_f1 = slot_f1
        self.value_precision = value_precision
        self.value_recall = value_recall
        self.value_f1 = value_f1
        self.pred_svids_with_no_match = no_match
        return pred_cluster_to_gold_slot_matches


