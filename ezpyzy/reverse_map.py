
def reverse_map(mapping):
    if isinstance(mapping, dict):
        rmap = {v: [] for v in mapping.values()}
        for k, v in mapping.items():
            rmap[v].append(k)
        return rmap
    else:
        raise ValueError("Expected a dictionary")