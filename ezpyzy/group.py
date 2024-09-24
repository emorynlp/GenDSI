
def group(items, key, sort=None):
    if callable(key):
        key = [key(item) for item in items]
    if not isinstance(key, list):
        key = [item[key] for item in items]
    assert len(key) == len(items), \
        "Length of 'key' list must match the length of 'items'"
    if sort is not None:
        if callable(sort):
            sort = {id(item): sort(item) for item in items}
        elif isinstance(sort, list):
            assert len(sort) == len(items), \
                "Length of 'sort' list must match the length of 'items'"
            sort = {id(item): sort[i] for i, item in enumerate(items)}
        else:
            sort = {id(item): item[sort] for item in items}
    grouped_dict = {group: [] for group in set(key)}
    for item, group in zip(items, key):
        grouped_dict[group].append(item)
    if isinstance(sort, dict):
        for group in grouped_dict:
            sortable = sorted((sort[id(item)], i, item) for i, item in enumerate(grouped_dict[group]))
            grouped = [item for _, _, item in sortable]
            grouped_dict[group] = grouped
    return grouped_dict