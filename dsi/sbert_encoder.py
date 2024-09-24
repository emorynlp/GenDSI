
from sentence_transformers import SentenceTransformer
import typing as T
import ezpyzy as ez
import torch as pt
import dsi.utils as diu


model = None


class SBERT:
    def __init__(self, encoding_type:str, batch_size:int=256):
        self.model_name = "all-MiniLM-L6-v2"
        self.encoding_type = encoding_type
        self.batch_size = batch_size

    @property
    def model(self):
        global model
        if model is None:
            model = SentenceTransformer(self.model_name)
        return model

    def encode(self,
        contexts: T.Iterable[str] = None,
        turns: T.Iterable[str] = None,
        slots: T.Iterable[str] = None,
        values: T.Iterable[str] = None,
    ):
        embeddings = []
        if self.encoding_type == 'ts':
            inputs = [f"{t}\n{s}:" for t, s in zip(turns, slots)]
        elif self.encoding_type == 'sv':
            inputs = [f'''The "{s}" is "{v}"''' for s, v in zip(slots, values)]
        elif self.encoding_type == 's':
            inputs = [f"{s}" for s in slots]
        elif self.encoding_type == 'v':
            inputs = [f"{v}" for v in values]
        else:
            raise Exception("Invalid encoding type")
        hash_content = inputs[0] + inputs[3] + inputs[-1] + str(len(inputs))
        content_hash = diu.non_stochastic_hash(hash_content)
        cache_path = f'cache/{self.model_name}_{self.encoding_type}_{content_hash}.pt'
        try:
            embeddings = pt.load(cache_path)
            return embeddings
        except FileNotFoundError:
            pass
        for batch in ez.batch(inputs, self.batch_size):
            embeddings.extend(self.model.encode(batch))
        embeddings = [pt.tensor(e, dtype=pt.float) for e in embeddings]
        stacked = pt.stack(embeddings)
        pt.save(stacked, cache_path)
        return embeddings


if __name__ == '__main__':
    sbert = SBERT(encoding_type='ts')
    tsv = [
        ("Hello can I have a pizza", "size", "large"),
        ("I'd like to order a medium pizza", "size", "medium"),
        ("Could I get a small pizza, please?", "size", "small"),
        ("Can I have a large pepperoni pizza?", "topping", "pepperoni"),
        ("I want a medium vegetarian pizza", "topping", "vegetarian"),
        ("Please add extra cheese to my small pizza", "topping", "extra cheese"),
        ("I'd like a large Hawaiian pizza", "style", "Hawaiian"),
        ("Can you make my medium pizza spicy?", "style", "spicy"),
        ("I prefer my small pizza to be gluten-free", "style", "gluten-free"),
        ("Could I have a large pizza with extra mushrooms?", "topping", "extra mushrooms")
    ]
    turns, slots, values = zip(*tsv)
    embeddings = sbert.encode(turns=turns, slots=slots)
    print(embeddings)
