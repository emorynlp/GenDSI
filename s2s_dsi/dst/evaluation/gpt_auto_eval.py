

from promptium.prompt import prompt # prompt magic
import dst.approaches.dst_seq_data as dstseq # data format
import dst.data.dst_data as dst # data format

# create some example data
data = dstseq.DstSeqData(dialogues=[
    dstseq.Dialogue(turns=[
        dstseq.Turn(
            turn="Hey there, I'm looking for a flight to New York.",
            slots={
                'greeting': ['yes'],
                'request': ['flight'],
                'destination': ['New York']
            },
            predicted_slots={
                'action': ['request flight'],
                'sentiment': ['positive']
            }
        ),
        dstseq.Turn(
            turn="Sure, we have several available every week. when would you like to fly?",
            slots={
                'flight available': ['yes'],
                'flight date': ['?']
            },
            predicted_slots={
                'time of departure': ['?'],
            }
        )
    ])
])

# create a prompt function with a prompt template
@prompt(model='gpt-3.5-turbo')
def predicted_slot_matches_reference(turn:dstseq.Turn, llm:callable=None):
    """
    Dialogue Context:
    Hi there, I would like 3 apples and nine bananas please, and I have this coupon. Do you have paper bags?

    Received Info:
    checking out: yes
    number of apples: 3
    has coupon: true
    paper bag available: ?

    Expected Info:
    greeting: yes
    action: purchase request
    items for purchase: 3 apples, 9 bananas
    coupon: yes
    paper bags: ?

    Expected Info Matches:
    greeting: yes -> no match
    action: purchase request -> checking out: yes
    items for purchase: 3 apples, 9 bananas -> no match
    coupon: yes -> has coupon: true
    paper bags: ? -> paper bag available: ?

    See above for an example of the task. Below are a Dialogue Context, and two sets of Info extracted from the Dialogue Context (Expected Info and Received Info). For each piece of Expected Info, match it to the corresponding piece of Received Info, or say "no match" if there is no match.

    Dialogue Context:
    {context}

    Received Info:
    {prediction}

    Expected Info:
    {reference}

    Expected Info Matches:

    """
    # preprocessing arguments to strings that can be fed into the prompt template
    context = '\n'.join(turn.turn for turn in turn.context())
    refs = {slot.name: slot for slot in turn.slots}
    preds = {slot.name: slot for slot in turn.predicted_slots}
    refstr = '\n'.join(f'{slot.name}: {", ".join(values)}' for slot, values in turn.slots.items() if values)
    predstr = '\n'.join(f'{slot.name}: {", ".join(values)}' for slot, values in turn.predicted_slots.items() if values)
    # calling GPT by filling the prompt template
    if preds and refs:
        generated = llm.generate(
            context=context,
            reference=refstr,
            prediction=predstr,
        )
    else:
        generated = ''
    # parsing the generated text
    ref_has_match:dict[tuple[dst.Turn, dst.Slot], bool] = {}
    pred_has_match:dict[tuple[dst.Turn, dst.Slot], bool] = {}
    lines = generated.split('\n')
    for line in lines:
        if '->' in line:
            ref, pred = line.split('->', 1)
            if ':' in ref and ':' in pred:
                ref_slot_name, ref_val = ref.split(':', 1)
                pred_slot_name, pred_val = pred.split(':', 1)
                ref_slot = refs.get(ref_slot_name.strip())
                pred_slot = preds.get(pred_slot_name.strip())
                if ref_slot and pred_slot:
                    ref_has_match[(turn, ref_slot)] = True
                    pred_has_match[(turn, pred_slot)] = True
    # the final product is this dictionary of bools--
    # whether each reference slot was matched by some predicted slot
    return ref_has_match, pred_has_match


# the main evaluation loop
def eval(data:dstseq.DstSeqData):
    all_matches = {}
    # iterate over turns
    for dialogue in data.dialogues:
        for turn in dialogue.turns:
            if turn.slots is not None:
                # call the prompt function
                matches = predicted_slot_matches_reference(turn, log=print, debug=True)
                # update the final dict of matches
                all_matches.update(matches)
                print(turn.turn)
                for slot in turn.slots:
                    if (turn, slot) in matches:
                        print(f'    ✅ {slot.name}')
                    else:
                        print(f'    ❌ {slot.name}')
    return all_matches

if __name__ == '__main__':
    matches = eval(data)
    x = 1



