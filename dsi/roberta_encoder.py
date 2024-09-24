from transformers import RobertaTokenizer, RobertaModel
import torch
import tqdm
import ezpyzy as ez
import typing as T
import dsi.globals as di_globals
import dsi.utils as diu


model = None


class RoBERTa:
    def __init__(self, encoding_type:str, batch_size: int=256, max_length: int=512):
        self.model_name = "roberta-base"
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.encoding_type = encoding_type
        self.batch_size = batch_size
        self.max_length = max_length
        self.model.eval()
        self.model.to('cuda')

    @property
    def model(self):
        global model
        if model is None:
            model = RobertaModel.from_pretrained(self.model_name)
        return model

    def encode(self,
        contexts: T.Iterable[str] = None,
        turns: T.Iterable[str] = None,
        slots: T.Iterable[str] = None,
        values: T.Iterable[str] = None,
    ):
        if not hasattr(turns, '__len__'):
            raise Exception("Turns must be provided")
        content = list(turns)
        hash_content = content[0]+content[3]+content[-1]+str(len(content))
        content_hash = diu.non_stochastic_hash(hash_content)
        cache_path = f'cache/{self.model_name}_{self.encoding_type}_{content_hash}.pt'
        try:
            embeddings = torch.load(cache_path)
            return embeddings
        except FileNotFoundError:
            pass
        if contexts is None:
            contexts = [None] * len(turns) # noqa
        embeddings = []
        pad, bos, eos = self.tokenizer.pad_token_id, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        context_strs = [f"{c}\n" if c else '' for c in contexts]
        turn_strs = [f"{t}\n" for t in turns]
        slot_strs = [f"{s}:" for s in slots]
        value_strs = [f"{v}" for v in values]
        context_tokens = self.tokenizer(
            context_strs, add_special_tokens=False, padding=False, truncation=False)['input_ids']
        turn_tokens = self.tokenizer(
            turn_strs, add_special_tokens=False, padding=False, truncation=False)['input_ids']
        slot_tokens = self.tokenizer(
            slot_strs, add_special_tokens=False, padding=False, truncation=False)['input_ids']
        value_tokens = self.tokenizer(
            value_strs, add_special_tokens=False, padding=False, truncation=False)['input_ids']
        batches = []
        for ctsv in ez.batch(list(zip(context_tokens, turn_tokens, slot_tokens, value_tokens)), self.batch_size):
            input_ids_batch = []
            attention_mask_batch = []
            boundaries_batch = []
            for c, t, s, v in ctsv:
                context, turn, slot, value = [self.tokenizer.decode(x) for x in [c, t, s, v]]
                length = 1 + len(c) + len(t) + len(s) + len(v) + 1
                if length > self.max_length:
                    c = c[length - self.max_length:]
                    length = 1 + len(c) + len(t) + len(s) + len(v) + 1
                if length > self.max_length:
                    raise Exception("Length of tokens exceeds maximum length, could not truncate context enough")
                pad_length = 512 - length
                p = [pad] * pad_length
                input_ids = torch.tensor([bos] + c + t + s + v + [eos] + p, dtype=torch.long)
                attention_mask = torch.cat((
                    torch.ones(length, dtype=torch.long), torch.zeros(pad_length, dtype=torch.long)))
                input_ids_batch.append(input_ids)
                attention_mask_batch.append(attention_mask)
                turn_i = 1 + len(c)
                slot_i = 1 + len(c) + len(t)
                value_i = 1 + len(c) + len(t) + len(s)
                end_i = 1 + len(c) + len(t) + len(s) + len(v)
                boundaries_batch.append((turn_i, slot_i, value_i, end_i))
            input_ids_batch = torch.stack(input_ids_batch)
            attention_mask_batch = torch.stack(attention_mask_batch)
            batches.append((dict(input_ids=input_ids_batch, attention_mask=attention_mask_batch), boundaries_batch))
        progress = tqdm.tqdm(total=len(turn_strs), desc="Encoding RoBERTa")
        for batch, boundaries in batches:
            with torch.no_grad():
                batch = {k: v.to('cuda') for k, v in batch.items()}
                outputs = self.model(**batch).last_hidden_state
            for i, (output, (turn_i, slot_i, value_i, end_i)) in enumerate(zip(outputs, boundaries)):
                all_tokens = self.tokenizer.decode(batch['input_ids'][i])
                if self.encoding_type == 's':
                    encoded_tokens = self.tokenizer.decode(batch['input_ids'][i][slot_i:value_i-1])
                    embeds = output[slot_i:value_i-1, :]
                    embedding = embeds.mean(dim=0)
                    embeddings.append(embedding)
                elif self.encoding_type == 'sv':
                    encoded_tokens = self.tokenizer.decode(batch['input_ids'][i][slot_i:end_i])
                    embeds = output[slot_i:end_i, :]
                    embedding = embeds.mean(dim=0)
                    embeddings.append(embedding)
                elif self.encoding_type == 'ts':
                    encoded_tokens = self.tokenizer.decode(batch['input_ids'][i][turn_i:value_i])
                    embeds = output[turn_i:value_i, :]
                    embedding = embeds.mean(dim=0)
                    embeddings.append(embedding)
                else:
                    raise Exception("Choose an encoding type from 's', 'sv', or 'ts'")
            progress.update(batch['input_ids'].shape[0])
        progress.close()
        stacked = torch.stack(embeddings)
        torch.save(stacked, cache_path)
        return embeddings


if __name__ == '__main__':
    roberta = RoBERTa(encoding_type='sv')
    tsv = [
        ("Please add extra cheese to my small pizza", "extra toppings", "extra cheese, extra mushroom, extra sauce"),
        ("Hello can I have a pizza", "size", "large"),
        ("I'd like to order a medium pizza", "size", "medium"),
        ("Could I get a small pizza, please?", "size", "small"),
        ("Can I have a large pepperoni pizza?", "topping", "pepperoni"),
        ("I want a medium vegetarian pizza", "topping", "vegetarian"),
        ("I'd like a large Hawaiian pizza", "style", "Hawaiian"),
        ("Can you make my medium pizza spicy?", "style", "spicy"),
        ("I prefer my small pizza to be gluten-free", "style", "gluten-free"),
        ("Could I have a large pizza with extra mushrooms?", "topping", "extra mushrooms")
    ]
    turns, slots, values = zip(*tsv)
    embeddings = roberta.encode(turns=turns, slots=slots, values=values)
    print(embeddings)