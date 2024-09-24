import json
import random as rng

import ezpyzy as ez
import dsi.cluster as dcl
import dsi.dst_data as dst
import difflib as dl
import typing as T

def replace(string: str, old: str|T.Iterable[str], new: str) -> str:
    if isinstance(old, str):
        return string.replace(old, new)
    else:
        for item in old:
            string = string.replace(item, new)
        return string


def evaluate_slot_value_discovery(data: dst.Data, output: dcl.Clustered, match:dict):
    domains = set(data.slots.domain)
    slots = {s for s in data.slots.slot if not any(d in s for d in [
        'bus', 'police'
    ])}
    values_by_slot = {}
    for slot, value in zip(data.slot_values.slot, data.slot_values.value):
        if slot in slots:
            values_by_slot.setdefault(slot, []).append(value)
    domain_agn_slots = {replace(slot, domains, '').strip() for slot in slots}
    values_domain_agn_by_slot = {}
    for slot, value in zip(data.slot_values.slot, data.slot_values.value):
        slot = replace(slot, domains, '').strip()
        if slot in domain_agn_slots:
            values_domain_agn_by_slot.setdefault(slot, []).append(value)
    tids_to_predicted_states = {}
    tids_to_actual_states = {}
    tids_to_accumulated = {}
    for dialogue in data.states():
        for state in dialogue:
            tids_to_actual_states[state.turn_id] = state.actual_state
            tids_to_predicted_states[state.turn_id] = state.predicted_state
            tids_to_accumulated[state.turn_id] = state.accumulation
    discoveries = {}
    true_discoveries = {}
    covered_gold_values = {}
    domain_agn_discoveries = {}
    domain_agn_true_discoveries = {}
    covered_domain_agn_values = {}
    for tid, cluster, discovered_value in zip(output.turn_id, output.cluster_id, output.value):
        slot = match.get(str(cluster))
        if slot in slots:
            discoveries[slot] = discoveries.get(slot, 0) + 1
            gold_accum = tids_to_accumulated.get(tid, {}).get(slot, [])
            # if not gold_accum:
            #     continue
            gold_values = values_by_slot.get(slot, [])
            matches = dl.get_close_matches(discovered_value, gold_values, cutoff=0.6)
            covered = [gv for gv, svid in zip(gold_values, gold_values) if gv in matches]
            covered_gold_values.setdefault(slot, []).extend(covered)
            if matches:
                true_discoveries[slot] = true_discoveries.get(slot, 0) + 1
        domain_agn_slot = replace(slot, domains, '').strip() if slot else None
        if domain_agn_slot in domain_agn_slots:
            domain_agn_discoveries[domain_agn_slot] = domain_agn_discoveries.get(domain_agn_slot, 0) + 1
            domain_agn_accum = {
                replace(slot, domains, '').strip(): accum
                for slot, accum in tids_to_accumulated.get(tid, {}).items()
            }.get(domain_agn_slot, [])
            # if not domain_agn_accum:
            #     continue
            gold_values = values_domain_agn_by_slot.get(domain_agn_slot, [])
            matches = dl.get_close_matches(discovered_value, gold_values, cutoff=0.6)
            covered = [gv for gv, svid in zip(gold_values, gold_values) if gv in matches]
            covered_domain_agn_values.setdefault(domain_agn_slot, []).extend(covered)
            if matches:
                domain_agn_true_discoveries[domain_agn_slot] = domain_agn_true_discoveries.get(domain_agn_slot, 0) + 1
    precisions = {}
    recalls = {}
    f1s = {}
    for slot in discoveries:
        precision = true_discoveries.get(slot, 0) / discoveries[slot]
        recall = len(set(covered_gold_values.get(slot, []))) / len(set(values_by_slot.get(slot, [])))
        f1 = 2 * (precision * recall) / (precision + recall)
        precisions[slot] = precision
        recalls[slot] = recall
        f1s[slot] = f1
    domain_agn_precisions = {}
    domain_agn_recalls = {}
    domain_agn_f1s = {}
    for slot in domain_agn_discoveries:
        precision = domain_agn_true_discoveries.get(slot, 0) / domain_agn_discoveries[slot]
        recall = len(set(covered_domain_agn_values.get(slot, []))) / len(set(values_domain_agn_by_slot.get(slot, [])))
        f1 = 2 * (precision * recall) / (precision + recall)
        domain_agn_precisions[slot] = precision
        domain_agn_recalls[slot] = recall
        domain_agn_f1s[slot] = f1
    results = {
        "discovery_precision": sum(precisions.values()) / len(precisions),
        "discovery_recall": sum(recalls.values()) / len(recalls),
        "discovery_f1": sum(f1s.values()) / len(f1s),
        "domain_agn_discovery_precision": sum(domain_agn_precisions.values()) / len(domain_agn_precisions),
        "domain_agn_discovery_recall": sum(domain_agn_recalls.values()) / len(domain_agn_recalls),
        "domain_agn_discovery_f1": sum(domain_agn_f1s.values()) / len(domain_agn_f1s),
    }
    return results

def evaluate_slot_induction(data_path, output_path, match_path, dsg_path):
    data = dst.Data(data_path)
    output = dcl.Clustered.of(output_path)
    slots = {s for s in data.slots.slot if not any(d in s for d in [
        'bus', 'police'
    ])}
    domains = set(data.slots.domain)
    domain_agn_slots = {replace(slot, domains, '').strip() for slot in slots}
    hand_eval = ez.File(match_path).load()
    clustermap = hand_eval['mapping']
    nonsense_cluster_labels = hand_eval['nonsense']
    num_induced_clusters = len(clustermap)
    num_ok_clusters = len([c for c in clustermap.values() if c not in nonsense_cluster_labels])
    domain_correct_clusters = []
    domain_agn_correct_clusters = []
    domain_agn_clusters = []
    intent_clusters = []
    status_clusters = []
    sentiment_clusters = []
    other_info_clusters = []
    nonsense_clusters = [c for c in clustermap.values() if c in nonsense_cluster_labels]
    cluster_counts = {}
    for cluster in clustermap.values():
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        if cluster in slots:
            domain_correct_clusters.append(cluster)
            domain_agn_correct_clusters.append(replace(cluster, domains, '').strip())
        elif cluster in domain_agn_slots:
            domain_agn_correct_clusters.append(cluster)
            domain_agn_clusters.append(cluster)
        elif 'intent' in cluster:
            intent_clusters.append(cluster)
        elif 'status' in cluster:
            status_clusters.append(cluster)
        elif 'sentiment' in cluster:
            sentiment_clusters.append(cluster)
        elif cluster not in nonsense_cluster_labels:
            other_info_clusters.append(cluster)
    redundant_clusters = list({c for c, count in cluster_counts.items() if count > 1})
    num_redundant_clusters = sum([count-1 for count in cluster_counts.values() if count > 1])
    missed_slots = list(slots - set(domain_correct_clusters))
    missed_domain_agn_slots = list(domain_agn_slots - set(domain_agn_correct_clusters))
    cluster_precision = (
        len(domain_correct_clusters) /
        (len(domain_correct_clusters) + len(nonsense_clusters))
    )
    cluster_recall = len(set(domain_correct_clusters)) / len(slots)
    cluster_f1 = 2 * (cluster_precision * cluster_recall) / (cluster_precision + cluster_recall)
    domain_agn_cluster_precision = (
        len(domain_agn_correct_clusters) /
        (len(domain_agn_correct_clusters) + len(nonsense_clusters))
    )
    domain_agn_cluster_recall = len(set(domain_agn_correct_clusters)) / len(domain_agn_slots)
    domain_agn_cluster_f1 = 2 * (domain_agn_cluster_precision * domain_agn_cluster_recall) / (
        domain_agn_cluster_precision + domain_agn_cluster_recall)
    error_analysis_svids = []
    for missed_slot in missed_slots:
        examples_with_missed = {
            svid for svid, slot in zip(
            data.slot_values.slot_value_id, data.slot_values.slot)
            if missed_slot == slot
        }
        examples_with_missed = rng.sample(list(examples_with_missed), 10)
        error_analysis_svids.extend(examples_with_missed)
    error_analysis_svs = ~data.slot_values[error_analysis_svids]
    ea_rows = error_analysis_svs.turn_id << data.turns.turn_id
    silver = dst.Data(dsg_path)
    silver_to_matched_slot = {}
    for svid, cluster in zip(output.slot_value_id, output.cluster_id):
        slot = clustermap.get(str(cluster))
        if slot:
            silver_to_matched_slot[svid] = slot
    tids_to_predicted_states = {}
    for dialogue in silver.states():
        for state in dialogue:
            tids_to_predicted_states[state.turn_id] = state.actual_state
    dsgs = [tids_to_predicted_states[tid] for tid in error_analysis_svs.turn_id]
    ea_table = ~ea_rows[ea_rows.text, ea_rows.slot, ea_rows.value]
    ea_table.pred = ez.Column(dsgs)
    ea_table.error = ez.Column([None] * len(ea_table))
    ea_table().save('results/mwoz_induction_error_analysis.csv')
    results = {
        "num_induced_clusters": num_induced_clusters,
        "num_ok_clusters": num_ok_clusters,
        "num_redundant_clusters": num_redundant_clusters,
        "cluster_precision": cluster_precision,
        "cluster_recall": cluster_recall,
        "cluster_f1": cluster_f1,
        "domain_agn_cluster_precision": domain_agn_cluster_precision,
        "domain_agn_cluster_recall": domain_agn_cluster_recall,
        "domain_agn_cluster_f1": domain_agn_cluster_f1,
        "correct_clusters": domain_correct_clusters,
        "domain_agn_clusters": domain_agn_clusters,
        "intent_clusters": intent_clusters,
        "status_clusters": status_clusters,
        "sentiment_clusters": sentiment_clusters,
        "other_info_clusters": other_info_clusters,
        "nonsense_clusters": nonsense_clusters,
        "redundant_clusters": redundant_clusters,
        "missed_slots": missed_slots,
        "missed_domain_agn_slots": missed_domain_agn_slots,
        "mapping": clustermap,
    }
    results.update(evaluate_slot_value_discovery(data, output, clustermap))
    return results
    
if __name__ == '__main__':
    results = evaluate_slot_induction(
        data_path='data/mwoz2.1/valid',
        output_path='data/mwoz2.1/mwoz_induction_output_ms_75.csv',
        match_path='results/mwoz_induction_hand_match_ms_75.json',
        dsg_path='data/silver_mwoz/valid'
    )
    with open('results/mwoz_induction_results_ms_75.json', 'w') as f:
        json.dump(results, f, indent=4)