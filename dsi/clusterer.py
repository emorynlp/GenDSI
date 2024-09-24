
import dataclasses as dc

import numpy as np

import ezpyzy as ez
import torch as pt
import dsi.globals as di_globals
from dsi.utils import cosine_similarity_matrix, euclidean_similarity_matrix
import dsi.utils as diu

if di_globals.accelerate:
    from cuml.cluster.dbscan import DBSCAN
    from cuml.cluster.hdbscan.hdbscan import HDBSCAN
else:
    from sklearn.cluster import DBSCAN
    from sklearn.cluster import HDBSCAN


@dc.dataclass
class Clusterer:
    algorithm: str
    eps: float = None
    min_samples: int = None
    min_cluster_size: int = None
    metric: str = 'cosine'
    merge_eps: float = 0.0
    max_cluster_size: int = None
    leaf_size: int = 35

    def __post_init__(self):
        if self.algorithm == "dbscan":
            self.clusterer = DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                leaf_size=self.leaf_size)
        elif self.algorithm == "hdbscan":
            self.clusterer = HDBSCAN(
                min_samples=self.min_samples,
                min_cluster_size=self.min_cluster_size,
                max_cluster_size=self.max_cluster_size or (0 if di_globals.accelerate else None),
                cluster_selection_epsilon=self.merge_eps,
                leaf_size=self.leaf_size,
                metric='euclidean')
        else:
            raise Exception("Invalid clustering algorithm")

    def cluster(self, embeddings):
        hash_content = '\n'.join(
            [
                f"algorithm: {self.algorithm}",
                f"eps: {self.eps}",
                f"min_samples: {self.min_samples}",
                f"min_cluster_size: {self.min_cluster_size}",
                f"metric: {self.metric}",
                f"merge_eps: {self.merge_eps}",
                f"max_cluster_size: {self.max_cluster_size}",
                f"{embeddings[0][0]}|{embeddings[0][1]}|",
                f"{embeddings[3][0]}|{embeddings[3][1]}|",
                f"{embeddings[-1][0]}|{embeddings[-1][1]}|",
                f"{embeddings[-1][-1]}|{embeddings[-1][-2]}|",
                f"{len(embeddings)}|{len(embeddings[0])}",
            ]
        )
        content_hash = diu.non_stochastic_hash(hash_content)
        if self.algorithm == "dbscan":
            cache_path = f'cache/dbscan_{self.eps}_{self.min_samples}_{content_hash}.json'
        else:
            cache_path = f'cache/hdbscan_{self.min_samples}_{self.min_cluster_size}_{self.max_cluster_size}_{self.merge_eps}_{content_hash}.json'
        cached_labels = ez.File(cache_path).load()
        if cached_labels:
            return cached_labels
        embeddings = np.stack(embeddings)
        with ez.Timer('Clustering...'):
            labels = self.clusterer.fit_predict(embeddings)
        if hasattr(labels, 'tolist'):
            labels = labels.tolist()
        ez.File(cache_path).save(labels)
        return labels


if __name__ == '__main__':
    points = [
        [0, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0.8, 0.1],
        [0, 0.1, 0.9],
        [1, 0, 0],
        [0.9, 0.1, 0.1],
        [1, 1, 1],
    ]
    vecs = [pt.tensor(x, dtype=pt.float32) for x in points]
    clusterer = Clusterer(algorithm='dbscan', eps=0.5, min_samples=2)
    labels = clusterer.cluster(vecs)
    print("\n".join(f"{p[0]:>5}{p[1]:>5}{p[2]:>5}   -> {l:>3}" for p, l in zip(points, labels)))




