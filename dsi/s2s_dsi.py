"""
Example usage of the dialogue state inference s2s model.

This bypasses the code used for experimentation because the experiment code (found in s2s_dsi folder) relies on loading in the dataset as a pickle object.
"""


import transformers as hf

device = 'cuda'

dsi = hf.AutoModelForSeq2SeqLM.from_pretrained(
    'jdfinch/dialogue_state_generator'
).to(device)

tokenizer = hf.AutoTokenizer.from_pretrained('t5-base')

def format_dialogue(turns: list[str]):
    context = [f"{s}: {t}" for s, t in reversed(tuple(zip("ABA", reversed(turns))))]
    return '\n'.join(['**', *context, '->'])

def infer_state(turns: list[str]):
    input = format_dialogue(turns)
    prompt = tokenizer(input, return_tensors='pt')['input_ids'].to(device)
    generation_config = hf.GenerationConfig(repetition_penalty=1.2, num_beams=5)
    generated_tokens, = dsi.generate(prompt, generation_config=generation_config, max_new_tokens=128)
    state_str = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    state = dict([x.strip() for x in sv.split(':', 1)] for sv in state_str.split('|') if ':' in sv)
    return state


dialogue = [
    "I am looking for an attraction in the city center.",
    "Ok, the broughton house gallery is in the centre and admission is free.",
]

state = infer_state(dialogue)

print(state)