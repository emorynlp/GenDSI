
import ezpyzy as ez
import random as rng
import dataclasses as dc
import dsi.dst_data as dst
import dsi.cluster as dcl
import dsi.preprocessing as dpp
import dsi.utils as diu
from dsi.sbert_encoder import SBERT
from dsi.roberta_encoder import RoBERTa
from dsi.clusterer import Clusterer
from dsi.reducer import Reducer
from collections import Counter
from tqdm import tqdm


@dc.dataclass
class InductorHyperparameters:
    encoding_model: str = None
    encoding_type: str = None
    clustering_algorithm: str = None
    dim_reduction_algorithm: str | None = None
    clustering_eps: float = None
    clustering_min_samples: int = None
    min_cluster_size: int = None
    cluster_merge_eps: float = 0.0
    max_cluster_size: int|None = None
    cluster_leaf_size: int = 40
    dim_reduction_n_neighbors: int | None = None
    dim_reduction_n_dimensions: int | None = None
    dim_reduction_min_dist: float | None = None
    exclude_speakers: tuple[str]|None = None
    exclude_values: tuple[str] = None
    filter_clusters_below_entropy: float = 0.0
    filter_clusters_with_prop_from_bot_turns: float = 0.0
    convert_booleans: bool = True
    filtered_dsg_model: bool = False



@dc.dataclass
class Inductor(InductorHyperparameters):
    def __post_init__(self):
        if self.encoding_model == "sbert":
            self.encoder = SBERT(encoding_type=self.encoding_type)
        elif self.encoding_model == "roberta":
            self.encoder = RoBERTa(encoding_type=self.encoding_type)
        else:
            raise Exception("Invalid encoder")
        self.eval_encoder = SBERT(encoding_type='sv')
        self.clusterer = Clusterer(
            algorithm=self.clustering_algorithm,
            eps=self.clustering_eps,
            min_samples=self.clustering_min_samples,
            min_cluster_size=self.min_cluster_size,
            merge_eps=self.cluster_merge_eps,
            max_cluster_size=self.max_cluster_size,
            leaf_size=self.cluster_leaf_size)
        if self.dim_reduction_algorithm:
            self.reducer = Reducer(
                n_neighbors=self.dim_reduction_n_neighbors,
                n_dimensions=self.dim_reduction_n_dimensions,
                min_dist=self.dim_reduction_min_dist)

    def encode_gold(self, data: dst.Data):
        if self.exclude_speakers:
            dpp.exclude_speakers(data, self.exclude_speakers)
        slots = dict.fromkeys(data.slots.slot)
        for i, s in enumerate(list(slots), 1):
            slots[s] = i
        did_to_turns = {}
        for text, did, tidx in zip(data.turns.text, data.turns.dialogue, data.turns.turn_index):
            if did not in did_to_turns:
                did_to_turns[did] = []
            while tidx >= len(did_to_turns[did]):
                did_to_turns[did].append('')
            did_to_turns[did][tidx] = text
        tid_to_context = {}
        for did, tidx, tid in zip(data.turns.dialogue, data.turns.turn_index, data.turns.turn_id):
            tid_to_context[tid] = '\n'.join(did_to_turns[did][:tidx])
        exs = data.slot_values.turn_id << data.turns.turn_id
        contexts = [tid_to_context[tid] for tid in exs.turn_id]
        encodings = self.eval_encoder.encode(
            contexts=contexts, turns=exs.text, slots=exs.slot, values=exs.value)
        cluster_column = [slots[s] for s in exs.slot]
        clustered_points = dcl.ClusteredPoint.of(dict(
            slot_value_id=exs.slot_value_id,
            slot=exs.slot,
            domain=exs.domain,
            value=exs.value,
            cluster_id=cluster_column,
            turn_id=exs.turn_id,
            point_encoding=encodings,
        ))
        return clustered_points

    def induce(self, data: dst.Data):
        if self.exclude_speakers:
            dpp.exclude_speakers(data, self.exclude_speakers)
        if self.exclude_values:
            svexs = data.slot_values
            mask = [v not in self.exclude_values for v in svexs.value]
            data.slot_values = ~svexs[mask]
        examples = data.slot_values.turn_id << data.turns.turn_id
        slots = examples.slot
        values = examples.value
        turns = examples.text
        turn_ids = examples.turn_id
        slot_value_ids = examples.slot_value_id
        encodings = self.eval_encoder.encode(turns=turns, slots=slots, values=values)
        if self.encoding_model != 'sbert' or self.encoding_type != self.eval_encoder.encoding_type:
            points = self.encoder.encode(turns=turns, slots=slots, values=values)
        else:
            points = encodings
        if self.dim_reduction_algorithm:
            points = self.reducer.reduce(points)
            self.clusterer.metric = 'euclidean'
        cluster_column = self.clusterer.cluster(points)
        if not self.exclude_speakers:
            dialogue_id_index_to_turn_id = {}
            for dialogue_id, index, turn_id in zip(
                examples.dialogue, examples.turn_index, examples.turn_id
            ):
                dialogue_id_index_to_turn_id[(dialogue_id, index)] = turn_id
            new_turn_ids = []
            for dialogue_id, index, turn_id, speaker in zip(
                examples.dialogue, examples.turn_index, examples.turn_id,
                examples.speaker
            ):
                if speaker == 'bot' and (dialogue_id, index + 1) in dialogue_id_index_to_turn_id:
                    new_turn_ids.append(dialogue_id_index_to_turn_id[(dialogue_id, index + 1)])
                else:
                    new_turn_ids.append(turn_id)
            turn_ids = new_turn_ids
        clustered_points = dcl.ClusteredPoint.of(dict(
            slot_value_id=slot_value_ids,
            slot=slots,
            value=values,
            cluster_id=cluster_column,
            turn_id=turn_ids,
            point_encoding=encodings,
        ))
        pred_names = {c: n for c, n in zip(clustered_points.cluster_id, clustered_points.slot)}
        values_by_cluster = {}
        for cid, value in zip(clustered_points.cluster_id, clustered_points.value):
            values_by_cluster.setdefault(cid, []).append(value)
        entropies_by_cluster = {cid: diu.entropy(values_by_cluster[cid]) for cid in values_by_cluster}
        low_entropy_filter = [entropies_by_cluster[cid] >= self.filter_clusters_below_entropy
            for cid in clustered_points.cluster_id]
        clustered_points = ~clustered_points[low_entropy_filter]
        if self.filter_clusters_with_prop_from_bot_turns:
            bot_turns = list(data.turns[data.turns.speaker == 'bot'].text)
            bot_text = '\n|\n'.join(bot_turns).lower()
            user_turns = list(data.turns[data.turns.speaker == 'user'].text)
            user_text = '\n|\n'.join(user_turns).lower()
            cluster_counts = Counter(clustered_points.cluster_id)
            in_bot_turns = Counter()
            for cid, value in tqdm(list(zip(clustered_points.cluster_id, clustered_points.value)), desc='filtering clusters from bot turn...'):
                if value.lower() not in user_text:
                    in_bot_turns[cid] += 1
            prop_in_bot_turns = {cid: in_bot_turns[cid] / cluster_counts[cid] for cid in cluster_counts}
            filtered_clusters = [prop_in_bot_turns[cid] < self.filter_clusters_with_prop_from_bot_turns
                for cid in clustered_points.cluster_id]
            clustered_points = ~clustered_points[filtered_clusters]
        if self.convert_booleans:
            clustered_points.value[:] = [v.lower() if isinstance(v, str) else v for v in clustered_points.value]
            clustered_points.value[:] = ['yes' if v == 'true' else v for v in clustered_points.value]
            clustered_points.value[:] = ['no' if v == 'false' else v for v in clustered_points.value]
        return clustered_points


def main():
    inductor = Inductor(
        encoding_model="sbert",
        encoding_type="sv",
        clustering_algorithm="hdbscan",
        dim_reduction_algorithm=None,
        clustering_eps=0.9,
        clustering_min_samples=3,
        min_cluster_size=3,
        max_cluster_size=None,
        cluster_merge_eps=0.00,
        dim_reduction_n_neighbors=10,
        dim_reduction_n_dimensions=10,
        dim_reduction_min_dist=0.1,
        exclude_speakers=None,
    )
    data = dst.Data('data/dsg5k/train')
    results = []
    partitions = dpp.partition_by_domain(data, 10)
    starting_cluster_id = 0
    for pid, partition in enumerate(partitions):
        induced = inductor.induce(partition)
        cluster_ids = [cid + starting_cluster_id if cid != -1 else cid for cid in induced.cluster_id]
        induced.cluster_id[:] = cluster_ids
        starting_cluster_id = max(cluster_ids) + 1
        results.extend(induced().dicts())
        samples = induced.samples(partition, n=4)
        print(
            samples[
                samples.cluster_id, samples.slot, samples.value, samples.text
            ]().display(max_cell_width=50)
        )
    results = dcl.ClusteredPoint.of(results)
    all = results.samples(data)
    all().save('data/dsg5k/dsg5k_induced.csv')
    samples = results.samples(data, n=10)
    print(
        samples[
            samples.cluster_id, samples.slot, samples.value, samples.text
        ]().display(max_cell_width=50))
    samples().save('results/dsg5k_induced_samples.csv', json_cells=False)

if __name__ == '__main__':
    main()
