import torch as pt
import dsi.globals as di_globals
import hashlib
import ezpyzy as ez
from tqdm import tqdm
import math
from collections import Counter
import itertools as it

def non_stochastic_hash(input_string):
    hash_object = hashlib.sha256()
    hash_object.update(input_string.encode('utf-8'))
    hash_digest = hash_object.hexdigest()
    return hash_digest


def cosine_similarity_matrix(tensors):
    if not isinstance(tensors, pt.Tensor):
        stacked_tensors = pt.stack(tensors)
    else:
        stacked_tensors = tensors
    if di_globals.accelerate:
        stacked_tensors = stacked_tensors.to('cuda')
    dot_products = []
    for batch in tqdm(list(ez.batch(stacked_tensors, 5000))):
        dot_product = pt.matmul(pt.stack(batch).to(
            'cuda' if di_globals.accelerate else 'cpu'), stacked_tensors.t())
        dot_products.append(dot_product.to('cpu'))
    del stacked_tensors
    dot_product = pt.cat(dot_products, dim=0).to('cuda' if di_globals.accelerate else 'cpu')
    return dot_product

def euclidean_similarity_matrix(tensors):
    if not isinstance(tensors, pt.Tensor):
        stacked_tensors = pt.stack(tensors)
    else:
        stacked_tensors = tensors
    if di_globals.accelerate:
        stacked_tensors = stacked_tensors.to('cuda')
    return pt.cdist(stacked_tensors, stacked_tensors, p=2)


def entropy(s):
    frequencies = Counter(s)
    total_length = len(s)
    probabilities = [float(freq) / total_length for freq in frequencies.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    return entropy


def roundrobin(*iterables):
    """Yields an item from each iterable, alternating between them.
        >>> list(roundrobin('ABC', 'D', 'EF'))
        ['A', 'D', 'E', 'B', 'F', 'C']
    """
    pending = len(iterables)
    nexts = it.cycle(iter(items).__next__ for items in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = it.cycle(it.islice(nexts, pending))


if __name__ == '__main__':
    input_string = "xx"
    result_entropy = entropy(input_string)
    print(f"Entropy of '{input_string}' is {result_entropy:.4f}")
