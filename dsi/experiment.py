
import itertools as it
import random
import ezpyzy as ez
import pathlib as pl
import dsi.dst_data as dst
import dsi.preprocessing as dpp
import dataclasses as dc
from dsi.inductor import Inductor, InductorHyperparameters
from dsi.matcher import Matcher
from dsi.sim_matcher import SimMatcher

mwoz_paths = dict(
    silver= 'data/silver_mwoz/valid',
    gold= 'data/mwoz2.1/valid',
)

sgd_paths = dict(
    silver= 'data/silver_sgd/valid',
    gold= 'data/sgd_wo_domains/valid',
)


mwoz_filtered_paths = dict(
    silver= 'data/silver_mwoz_filtered/valid',
    gold= 'data/mwoz2.1/valid',
)

sgd_filtered_paths = dict(
    silver= 'data/silver_sgd_filtered/valid',
    gold= 'data/sgd_wo_domains/valid',
)

data_paths = sgd_filtered_paths

hyperparam_search_space = dict(
    clustering_eps = [None],
    clustering_min_samples = [5],
    min_cluster_size = [25],
    cluster_merge_eps = [0.3],
    max_cluster_size = [None],
    dim_reduction_n_neighbors = [None],
    dim_reduction_n_dimensions = [None],
    dim_reduction_min_dist = [None],
)

approach_hyperparams = {
    "sbert": (),
    "roberta": (),
    "dbscan": ("clustering_eps", "clustering_min_samples", "min_cluster_size"),
    "hdbscan": ("min_cluster_size", "cluster_merge_eps", "max_cluster_size", "clustering_min_samples"),
    "umap": ("dim_reduction_n_neighbors", "dim_reduction_n_dimensions", "dim_reduction_min_dist"),
}

approaches = dict(
    encoding_model = ["sbert"],
    encoding_type = ["sv",],
    clustering_algorithm = ["hdbscan"],
    dim_reduction_algorithm = [None],
    exclude_speakers = [None],
    exclude_values = [("?",)],
    filter_clusters_below_entropy = [0.0],
    filter_clusters_with_prop_from_bot_turns = [0.5],
    filtered_dsg_model = [bool('filtered' in data_paths['silver'])],
)

def search():
    approach_candidates = list(it.product(*approaches.values()))
    for approach_values in approach_candidates:
        approach = dict(zip(approaches, approach_values))
        approach_hypers = {x for a in approach.values() for x in approach_hyperparams.get(a, ())}
        search_space = {
            k:v for k,v in hyperparam_search_space.items() if k in approach_hypers}
        for hyperparam_values in it.product(*search_space.values()):
            hyperparams = dict(zip(search_space, hyperparam_values))
            hyperparams.update(approach)
            yield hyperparams

def experiment(gold_data, silver_data, hyperparams,
    max_clusters_for_display=1_000,
    min_clusters_for_display=10,
    num_examples_for_cluster_display=10,
):
    print("="*80, '\n')
    print("\n".join(f"{k}: {v}" for k,v in hyperparams.items()))
    inductor = Inductor(**hyperparams)
    gold_clustered_points = inductor.encode_gold(gold_data)
    pred_clustered_points = inductor.induce(silver_data)
    print(f"Gold clusters: {len(set(gold_clustered_points.cluster_id))}")
    print(f"Pred clusters: {len(set(pred_clustered_points.cluster_id))}")
    matcher = SimMatcher()
    matching = matcher.match_values(gold_clustered_points, pred_clustered_points)
    gold_cluster_to_name = {c: n for c, n in zip(gold_clustered_points.cluster_id, gold_clustered_points.slot)}
    named_matching = {p: gold_cluster_to_name.get(g) for p, g in matching.items()}
    all_examples = pred_clustered_points.samples(silver_data, gold_data, matches=named_matching)
    samples = pred_clustered_points.samples(silver_data, gold_data, matches=named_matching, n=num_examples_for_cluster_display)
    samples().sort(samples.cluster_id)
    samples().save(f'results/induction_{data_paths["gold"].replace("/", "_")}_samples.csv', json_cells=False)
    all_examples().sort(all_examples.cluster_id)
    all_examples().save(f'results/induction_{data_paths["gold"].replace("/", "_")}_output.csv')
    print(samples[samples.cluster_id, samples.matched_slot, samples.slot, samples.value, samples.text]().display(max_cell_width=50))
    noise = [x for x in pred_clustered_points.cluster_id if x == -1]
    print(f"Noise: {len(noise)} / {len(pred_clustered_points.cluster_id)}")
    print(f"Value precision: {matcher.value_precision}")
    print(f"Value recall: {matcher.value_recall}")
    print(f"Value F1: {matcher.value_f1}")
    print(f"Precision: {matcher.cluster_precision}")
    print(f"Recall: {matcher.cluster_recall}")
    print(f"F1: {matcher.cluster_f1}")
    return matcher

def main(max_clusters_for_display=1_000):
    gold = dst.Data(data_paths['gold'])
    silver = dst.Data(data_paths['silver'])
    # random.seed(42)
    # gold = dpp.downsample_examples(gold, 1000)
    # silver = dpp.downsample_examples(silver, 1000)
    search_space = list(search())
    print("Search space size:", len(search_space), "\n")
    results = pl.Path(f'results/induction_{data_paths["gold"].replace("/","_")}.csv')
    all_hypers_names = [f.name for f in dc.fields(InductorHyperparameters)] + ["precision", "recall", "f1", "N", "v_precision", "v_recall", "v_f1"]
    if not results.exists():
        results.write_text(",".join(all_hypers_names))
    file = ez.File(results)
    random.shuffle(search_space)
    for i, hyperparams in enumerate(search_space):
        matching = experiment(gold, silver, hyperparams,
            max_clusters_for_display=max_clusters_for_display)
        print(f"\nDone {i+1}/{len(search_space)} experiments\n")
        result = {
            **{n: str(hyperparams.get(n,'')).replace(',',' ') for n in all_hypers_names},
            "precision": matching.cluster_precision,
            "recall": matching.cluster_recall,
            "f1": matching.cluster_f1,
            "N": matching.n,
            "v_precision": matching.value_precision,
            "v_recall": matching.value_recall,
            "v_f1": matching.value_f1,
        }
        file.append("\n" + ",".join(str(x) for x in result.values()))
        # break


if __name__ == "__main__":
    for dp in (sgd_paths,):
        data_paths = dp
        main()




