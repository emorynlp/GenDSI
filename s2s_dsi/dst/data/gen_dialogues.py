import regex

from promptium.prompt import prompt
import promptium.parse as parse
import promptium.gpt as gptapi
import dst.data.parse as dst_parse
import pickle
import json
import sentence_transformers as st
import sentence_transformers.util as stu
import pathlib

def gen_lotsa_tasks(n, **ops):
    file_path = pathlib.Path('llm_cache/lotsa_tasks.json')
    model = st.SentenceTransformer('all-MiniLM-L6-v2')
    if file_path.exists():
        with open(file_path) as f:
            tasks = json.load(f)
    else:
        tasks = []
    while len(tasks) < n:
        updated = []
        if tasks:
            ops.update(gen_recache='lotsa_tasks')
        new_tasks = gen_tasks(100, **ops)
        if not isinstance(new_tasks, list):
            continue
        all_tasks = tasks + new_tasks
        embeddings = model.encode(all_tasks, convert_to_tensor=True)
        groups = stu.community_detection(
            embeddings, threshold=0.80, min_community_size=1,
        )
        for i, group in enumerate(reversed(groups)):
            print(f'Group {i}:')
            print('\n'.join([f'  {all_tasks[idx]}' for idx in group]))
            selection = all_tasks[sorted(group)[-1]]
            updated.append(selection)
        with open(file_path, 'w') as f:
            json.dump(updated, f)
        tasks = updated


@prompt
def gen_tasks(n, generated=None):
    """
    List {n} diverse examples of everyday tasks that require talking to another person. Format each list item like:

    N. <Role of person 1> talks to <role of person 2> in order to <task goal>


    """
    scenarios = parse.parse(generated, parse.list_items)
    return scenarios

@prompt
def gen_ontology(task, idx=0, generated=None):
    """
    List examples of as many different types of information as you can that would be shared during the dialogue scenario: {task}


    """
    return generated

@prompt
def gen_dialogue(task, ontology, generated=None):
    """
    Dialogue Scenario:
    {task}

    Information Types:
    {ontology}

    Write a dialogue for the above Dialogue Scenario. Include specific examples of the Information Types above being shared and implied throughout the conversation. Make up actual names/values when specific information examples are shared.


    """
    return parse.parse(generated, parse.label_items)

@prompt
def gen_extract(ontology, dialogue, generated=None):
    """
    Extract as many variables as you can from the Dialogue, formatted in JSON (use "?" if a value is being requested), like

    {
        "<variable name>": <value>,
        ...
    }

    Dialogue:
    {dialogue}


    """
    return dst_parse.parse(generated), dialogue


def gen_pipeline(**ops):
    path = pathlib.Path('llm_cache', 'gpt10k')
    tasks = json.loads((path / 'tasks.json').read_text())
    print(f'Generating {len(tasks)} tasks')
    for task in tasks:
        ontology = gen_ontology(task, **ops)
        dialogue = gen_dialogue(task, ontology, **ops)
        for i in range(2, len(dialogue)):
            history = dialogue[max(i - 3, 0):i]
            context = '\n'.join([f'{s}: {t}' for s, t in history])
            gen_extract(ontology, context, **ops)


import multiprocessing as mp

def gen_sub_pipeline(tasks, n_per_task, k, cache_folder='llm_cache', ops=None):
    if ops is None:
        ops = {}
    ops.update(dict(
        cache_folder=f'{cache_folder}/{k}th_proc'
    ))
    for j, task in enumerate(tasks):
        for m in range(n_per_task):
            ontology = gen_ontology(task, m, gen_recache=True, **ops)
            dialogue = gen_dialogue(task, ontology, **ops)
            for i in range(2, len(dialogue)):
                history = dialogue[max(i - 3, 0):i]
                context = '\n'.join([f'{s}: {t}' for s, t in history])
                gen_extract(ontology, context, **ops)
            print(f'Process {k} completed {j+1}/{len(tasks)} tasks')
            print(
                '   ',
                f'{gptapi.tokens:,} tokens used in {gptapi.runtime()/60:.1f} min',
                flush=True
            )

def gen_multi_pipeline(tasks, n_per_task, procs, cache_folder='llm_cache', **ops):
    tasks = [tasks[i::procs] for i in range(procs)]
    with mp.Pool(procs) as pool:
        pool.starmap(
            gen_sub_pipeline,
            [(tasks[i], n_per_task, i, cache_folder, ops) for i in range(procs)]
        )


####################################################

def collect_multi_pipeline_results(folder):
    path = pathlib.Path(folder)
    ontology_gens = {}
    dialogue_gens = {}
    extract_gens = {}
    for proc_folder in path.glob('*th_proc'):
        ontology_path = proc_folder / 'gen_ontology.gen.json'
        dialogue_path = proc_folder / 'gen_dialogue.gen.json'
        extract_path = proc_folder / 'gen_extract.gen.json'
        ontology_dump = ontology_path.read_text()
        dialogue_dump = dialogue_path.read_text()
        extract_dump = extract_path.read_text()
        ontology = json.loads(ontology_dump)
        dialogue = json.loads(dialogue_dump)
        extract = json.loads(extract_dump)
        ontology_gens.update(ontology)
        dialogue_gens.update(dialogue)
        extract_gens.update(extract)
    ontology_path = path / 'gen_ontology.gen.json'
    dialogue_path = path / 'gen_dialogue.gen.json'
    extract_path = path / 'gen_extract.gen.json'
    ontology_path.write_text(json.dumps(ontology_gens))
    dialogue_path.write_text(json.dumps(dialogue_gens))
    extract_path.write_text(json.dumps(extract_gens))
    print(f'Ontologies: {len(ontology_gens)}')
    print(f'Dialogues: {len(dialogue_gens)}')
    print(f'Extracts: {len(extract_gens)}')
    ontology_gens = {json.loads(k): v for k, v in ontology_gens.items()}
    dialogue_gens = {json.loads(k): v for k, v in dialogue_gens.items()}
    extract_gens = {json.loads(k): v for k, v in extract_gens.items()}
    ontologies = {} # {scenario: ontology}
    dialogues = {}  # {dialogue: (scenario, ontology)}
    extracts = {}   # {dialogue_text: extract}
    scenarios = {}  # {task: {dialogue: {context: extract}}}
    for prompt, gen in ontology_gens.items():
        scenario = regex.match(r'"?List .*?: (.*)', prompt).group(1).strip()
        ontology = gen.strip()
        ontologies[scenario] = ontology
    def remove_pre_speaker_noise(dialogue):
        d = []
        for s, t in dialogue:
            if '\n' in s:
                s = s.split('\n')[-1]
            d.append((s, t))
        return tuple(d)
    for prompt, gen in dialogue_gens.items():
        state = 'preamble'
        scenario = []
        ontology = []
        prompt_lines = prompt.split('\n')
        for line in prompt_lines:
            if 'Dialogue Scenario:' in line:
                state = 'scenario'
            elif 'Information Types:' in line:
                state = 'ontology'
            elif 'Write a dialogue for the above Dialogue Scenario.' in line:
                break
            elif state == 'scenario':
                scenario.append(line)
            elif state == 'ontology':
                ontology.append(line)
        scenario = '\n'.join(scenario).strip()
        ontology = '\n'.join(ontology).strip()
        dialogue = parse.parse(gen, parse.label_items)
        dialogue = tuple((speaker.strip(), turn.strip()) for speaker, turn in dialogue)
        dialogue = remove_pre_speaker_noise(dialogue)
        if dialogue:
            dialogues[dialogue] = (scenario, ontology)
    for prompt, gen in extract_gens.items():
        state = 'preamble'
        dialogue = []
        prompt_lines = prompt.split('\n')
        for line in prompt_lines:
            if 'Dialogue:' in line:
                state = 'dialogue'
            elif state == 'dialogue':
                if line:
                    dialogue.append(line)
        dialogue_turns = []
        for line in dialogue:
            dialogue_turns.append(line)
        dialogue_text = '\n'.join(dialogue).strip()
        dialogue = parse.parse(dialogue_text, parse.label_items)
        dialogue = tuple((speaker.strip(), turn.strip()) for speaker, turn in dialogue)
        dialogue = remove_pre_speaker_noise(dialogue)
        extract = dst_parse.parse(gen)
        extracts[dialogue] = extract
    for dialogue, (scenario, ontology) in dialogues.items():
        scenarios.setdefault(scenario, {})[dialogue] = {}
    context_caster = lambda c: tuple(
        (s.replace('\n', ''), t.replace('\n', '')) for s, t in c
    )
    contexts = {}
    for dialogue in dialogues:
        for i in range(2, len(dialogue)):
            context = dialogue[max(i - 3, 0):i]
            key = context_caster(context)
            contexts[key] = dialogue
    for context, extract in extracts.items():
        key = context_caster(context)
        dialogue = contexts[key]
        scenario, ontology = dialogues[dialogue]
        scenarios[scenario][dialogue][context] = extract
    with open('llm_cache/gpt10k/gpt10k.pkl', 'wb') as f:
        pickle.dump(scenarios, f)
    print(f'Scenarios: {len(scenarios)}')
    num_dialogues = sum(len(v) for v in scenarios.values())
    print(f'Dialogues: {num_dialogues}')
    num_contexts = sum(
        sum(len(v) for v in d.values()) for d in scenarios.values()
    )
    print(f'Contexts: {num_contexts}')


@prompt
def fix_extraction(dialogue, extractions, llm=None):
    """
    Your job is to identify as many specific variables as you can from a dialogue response.
    As a reference for the proper kind of format and content of the variables you will extract, here are some examples of variables extracted from other similar dialogues (note that these have some errors in them):

    {extractions}

    Here are the previous dialogue turns (do not identify variables in these, unless the information also appears in the response):
.
    {context}

    Here is the dialogue response:

    "{response}"

    Write as many variables and values as you can find in the dialogue response. Make the variables and corresponding values you find are newline separated in the format:
    <variable>: <value>



    :param dialogue:
    :param extractions:
    :param llm:
    :return:
    """
    # try some different prompts
    context = '\n'.join(f'"{t}"' for s, t in dialogue[:-1])
    response = f'{dialogue[-1][1]}'
    extractions = '\n'.join(f'{k}: {v}' for extracts in extractions for k, v in extracts.items())
    if not extractions:
        ... # use some default example for the extractions, since there were none in the original example
            # could also add a parameter to this function for a default extraction example and pass in a random extractionf from the same scenario to the default
            # use the next non-empty extraction from the same dialogue as the default
    gen = llm.generate(extractions, context, response)
    parsed_output = parse.parse(gen, parse.label_items)
    return parsed_output


'''
Additional prompt ideas:

1. "Here is an erroneous extraction, please fix it"
2. "Select the subset of the extraction that is directly relevant to the response"
3. ...

'''

import random

def align_multi_pipeline_results(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)  # {scenario: {dialogue: {context: extract}}}
    data = list(data.items())
    random.shuffle(data)
    examples_done = 0
    for scenario, dialogue_data in data:
        for i, (dialogue, turn_data) in enumerate(dialogue_data.items()):
            extractions = [[] for _ in dialogue] # list of list of extractions for each turn
            for j, (context, extract) in enumerate(turn_data.items()):
                start_turn_index = None
                for k, (s, t) in enumerate(dialogue):
                    if context[0][1] == t:
                        start_turn_index = k
                        break
                end_turn_index = start_turn_index + len(context)
                extract = dst_parse.flatten(extract)
                extract = dst_parse.clean_example(dialogue, extract, end_turn_index-1)
                for k in range(start_turn_index, end_turn_index):
                    extractions[k].append(extract)
            for j, ((s, t), extracts) in enumerate(zip(dialogue, extractions)):
                dialogue_context = dialogue[max(0, j-2):j+1]
                current_turn = dialogue_context[-1]
                new_extracts = fix_extraction(
                    dialogue_context,
                    extracts,
                    log=print,
                    debug=True,
                    recache=True,
                )
                examples_done += 1
                if examples_done >= 100:
                    return




if __name__ == '__main__':

    # collect_multi_pipeline_results('llm_cache/gpt10k')
    align_multi_pipeline_results('llm_cache/gpt10k/gpt10k.pkl')


