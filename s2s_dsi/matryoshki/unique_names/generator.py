import os
import random
import string

def remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

def get_random_name(existing_names=None, tries=10 ** 5):
    existing_names = set(existing_names) if existing_names is not None else None
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    adj = list({remove_punctuation(x[x.find('.')+1:].strip().replace(' ', '')) for x in open(f"{current_file_path}/adjectives.txt").readlines()})
    nouns = list({remove_punctuation(x[x.find('.')+1:].strip().replace(' ', '')) for x in open(f"{current_file_path}/star_wars_nouns.txt").readlines()})
    if existing_names is None:
        return f"{random.choice(adj)}{random.choice(nouns)}"
    else:
        for i in range(tries):
            gen = f"{random.choice(adj).lower()}-{random.choice(nouns).lower()}"
            if gen not in existing_names:
                return gen
    newline = '\n'
    raise ValueError(f'Cannot generate a unique name for model! Existing model names: {newline.join(existing_names)}')



if __name__ == '__main__':
    collisions = set()
    for _ in range(1000):
        gen = get_random_name(collisions)
        print(gen)
        collisions.add(gen)
