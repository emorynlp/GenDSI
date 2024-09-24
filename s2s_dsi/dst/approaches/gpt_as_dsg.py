
import dst.data.gpt_generate_data as ggd
import dst.approaches.dst_seq_data as dsd
import dst.data.dst_data as dst
import dst.data.gptdst_to_dst_format as gtdf
import ezpyz as ez
import traceback


def convert_dialogues_to_gpt_pipeline_format(
    data: dsd.DstSeqData,
    scenario_prefix = '',
) -> list[ggd.Dialogue]:
    dialogues = []
    for dial in data.dialogues:
        scenario = scenario_prefix + ', '.join(sorted(dial.domains()))
        dialogue = ggd.Dialogue(scenario=scenario)
        for i, turn in enumerate(dial.turns):
            text = turn.turn
            dialogue.speakers.add(turn.speaker)
            turn_index = len(dialogue.turns)
            dialogue.turns.append(ggd.Turn(
                text=text,
                dialogue=dialogue,
                index=turn_index,
                speaker=turn.speaker,
            ))
        dialogues.append(dialogue)
    return dialogues


def rematch_outputted_slots(
    original_data: dsd.DstSeqData,
    outputted_data: list[ggd.Dialogue],
):
    for dialogue, dial_out in zip(original_data.dialogues, outputted_data):
        for turn, turn_out in zip(dialogue.turns, dial_out.turns):
            turn: dsd.Turn
            if turn_out.example:
                turn.predicted_slots = {}
                out_slots = turn_out.example.slots
                for out_slot_name, out_slot in out_slots.items():
                    out_slot: ggd.Slot
                    out_value = out_slot.value
                    slot_name = gtdf.postprocess_slot_name(out_slot_name)
                    slot_value = gtdf.postprocess_slot_value(out_value, None, None, None) # noqa
                    slot = dst.Slot(name=slot_name)
                    turn.predicted_slots[slot] = slot_value
                # turn_text = turn.turn
                # prediction = {slot.name: ', '.join(values) for slot, values in turn.predicted_slots.items()}
                # ...

def gpt_as_dsg(
    data: dsd.DstSeqData|str,
    scenario_prefix = '',
    num_examples = None,
    num_dialogues = None,
    save_path = None,
    seed=42,
):
    if isinstance(data, str):
        data = dsd.DstSeqData.load(data)
    dialogues = convert_dialogues_to_gpt_pipeline_format(data, scenario_prefix)
    ggd.gen_dst_data(
        dialogues,
        num_examples=num_examples,
        num_dialogues=num_dialogues,
        just_qa_pairs=False,
        include_slot_description=False,
        seed=seed, log=print
    )
    rematch_outputted_slots(data, dialogues)
    if save_path is not None:
        data.save(save_path)
    return data

def convert_train_to_predictions(data:ez.filelike, save_path=None):
    data = dsd.DstSeqData.load(data)
    for dialogue in data.dialogues:
        for turn in dialogue.turns:
            turn.predicted_slots = turn.slots
    if save_path:
        data.save(save_path)
    return data


def main():
    try:
        gpt_as_dsg(
            'data/sgd-data/sgd_100_test.pkl',
            "A user talks with an assistant about the topics/services: ",
            num_examples=None,
            num_dialogues=None,
            save_path='ex/Seq2seqDSG/GPTpipeline/1/predictions/sgd_100_test.dstseqdata',
            seed=42,
        )
    except Exception:
        traceback.print_exc()
        ez.email(
            'jamesfinch293@gmail.com', 'GPT as DSG error', traceback.format_exc()
        )
    else:
        print('Done!!!!!!!!!!')
        ez.email(
            'jamesfinch293@gmail.com', 'GPT as DSG done', 'No errors'
        )



if __name__ == '__main__':

    main()

"""
conda activate dstr
export PYTHONPATH=/local/scratch/jdfinch/dstrr/:/local/scratch/jdfinch/dstrr/src/
nohup python -u src/dst/approaches/gpt_as_dsg.py > gpt_as_dsg.log &
"""