from transformers import RobertaTokenizer, RobertaModel
import torch

# Load pre-trained model tokenizer (vocabulary)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


dialogue = "This is a dialogue"
slot = "quantity"

dialogue_ids = tokenizer.encode(dialogue, add_special_tokens=True, return_tensors='pt')
slot_ids = tokenizer.encode(slot, add_special_tokens=True, return_tensors='pt')
slot_position = len(dialogue_ids['input_ids'])

all_ids = torch.cat((dialogue_ids['input_ids'], slot_ids['input_ids']), dim=1)
input_ids = dict(input_ids=all_ids, attention_mask=torch.ones(all_ids.shape))


# Encode text
# text = "Hello this is a test case please encode me"
# input_ids = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')
# """input_ids
# dict looks like {
#     'input_ids': tensor([[  0, 31414,  232,   328,   16,   10...
#     'attention_mask': tensor([[0, 1, 1, ...
# }
# """

# Load pre-trained model (weights)
model = RobertaModel.from_pretrained('roberta-base')

# Set the model to evaluation mode
model.eval()

# Forward pass, get embeddings
with torch.no_grad():
    outputs = model(input_ids)

# Retrieve the hidden states (last layer)
last_hidden_states = outputs.last_hidden_state

# Convert the tensor to a list of Python lists
embeddings = last_hidden_states.squeeze(0).tolist()

token_embedding_list = []
for token_id, embedding in zip(input_ids['input_ids'], embeddings):
    token = tokenizer.decode(token_id)
    token_embedding_list.append((token, embedding))

# some algorithm for just getting the slot name part

# Print token embeddings
for token, embedding in zip(tokenizer.tokenize(text), embeddings):
    print(token, embedding)