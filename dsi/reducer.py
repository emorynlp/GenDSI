
import dataclasses as dc
import dsi.globals as di_globals
import torch as pt
import dsi.utils as diu
import ezpyzy as ez

if di_globals.accelerate:
    import cuml
    import cudf
    UMAP = cuml.UMAP # noqa
else:
    import umap
    UMAP = umap.UMAP



@dc.dataclass
class Reducer:
    n_neighbors: int
    n_dimensions: int
    min_dist: float
    metric: str = 'cosine'

    def __post_init__(self):
        self.reducer = UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_dimensions,
            min_dist=self.min_dist,
            metric=self.metric,
        )

    def reduce(self, embeddings):
        if not isinstance(embeddings, pt.Tensor):
            array = pt.stack(embeddings).cpu().numpy()
        else:
            array = embeddings.cpu().numpy()
        content = str(array)
        content_hash = diu.non_stochastic_hash(content)
        cache_path = f'cache/umap_{self.n_neighbors}_{self.n_dimensions}_{self.min_dist}__{self.metric}_{content_hash}.pt'
        try:
            embeddings = pt.load(cache_path)
            return embeddings
        except FileNotFoundError:
            pass
        with ez.Timer('dim reduction...'):
            if di_globals.accelerate:
                embeddings = cudf.DataFrame(array) # noqa
                embeddings = self.reducer.fit_transform(embeddings)
                embeddings = embeddings.to_pandas().values
                embeddings = pt.tensor(embeddings, dtype=pt.float32)
                pt.save(embeddings, cache_path)
                return embeddings
            else:
                embeddings = self.reducer.fit_transform(array) # noqa
                embeddings = pt.tensor(embeddings, dtype=pt.float32)
                pt.save(embeddings, cache_path)
                return embeddings



if __name__ == '__main__':
    points = [
        [0, 0, 1, 0.2, 0.5, 0.7, 0.3, 0.9],
        [0, 1, 0, 0.4, 0.6, 0.1, 0.8, 0.2],
        [0, 0, 1, 0.6, 0.3, 0.2, 0.5, 0.7],
        [0, 0.8, 0.1, 0.9, 0.4, 0.3, 0.6, 0.2],
        [0, 0.1, 0.9, 0.3, 0.8, 0.5, 0.1, 0.4],
        [1, 0, 0, 0.7, 0.2, 0.9, 0.4, 0.6],
        [0.9, 0.1, 0.1, 0.5, 0.7, 0.8, 0.2, 0.3],
        [1, 1, 1, 0.8, 0.6, 0.4, 0.7, 0.1]
    ]
    vecs = [pt.tensor(x, dtype=pt.float32) for x in points]
    reducer = Reducer(n_neighbors=2, n_dimensions=2, min_dist=0.1)
    reduced = reducer.reduce(vecs)
    import matplotlib.pyplot as plt
    plt.scatter(reduced[:, 0], reduced[:, 1])
    plt.title('UMAP Reduced Points')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()