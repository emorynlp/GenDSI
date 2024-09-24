import dataclasses

import dst.data.dst_data as dst
import dst.approaches.sequences_data as seq
import ezpyz as ez


class Turn(dst.Turn):
    def __init__(
        self,
        turn: dst.turnlike,
        speaker=None,
        listener=None,
        slots: dict[dst.slotlike, str | list[str]] = None,
        predicted_slots: dict[dst.slotlike, str | list[str]] = None,
        dialogue=None,
        index=None
    ):
        super().__init__(
            turn=turn,
            speaker=speaker,
            listener=listener,
            slots=slots,
            predicted_slots=predicted_slots,
            dialogue=dialogue,
            index=index
        )
        self.slot_sequences:dict[dst.Slot, seq.Sequence]|None = {
            slot: seq.Sequence() for slot in self.slots
        } if self.slots is not None else None
        self.state_sequence: seq.Sequence | None = None


class Dialogue(dst.Dialogue):
    Turn = Turn
    turns: list[Turn]

class DstSeqData(dst.DstData):
    Dialogue = Dialogue
    dialogues: list[Dialogue]

    @classmethod
    def load(cls, file:ez.filelike, *args, **kwargs) ->'DstSeqData':
        loaded = super().load(file, *args, **kwargs)
        casted = DstSeqData(loaded)
        return casted

    def add(self, dialogue:dst.dialoguelike):
        dialogue = super().add(dialogue)
        for turn in dialogue.turns:
            if turn.slot_sequences:
                for slot, seq in list(turn.slot_sequences.items()):
                    del turn.slot_sequences[slot]
                    slot = self.ontology.add(slot)
                    turn.slot_sequences[slot] = seq

    def slot_sequences(self):
        sequences = seq.SequencesData([
            sequence
            for dialogue in self.dialogues
            for turn in dialogue.turns if turn.slots is not None
            for slot, sequence in turn.slot_sequences.items()
        ])
        return sequences

    def state_sequences(self):
        sequences = seq.SequencesData([
            turn.state_sequence
            for dialogue in self.dialogues
            for turn in dialogue.turns if turn.state_sequence is not None
        ])
        return sequences

@dataclasses.dataclass
class DstSeqResults(seq.SequenceResults, dst.DstResults):
    pass


if __name__ == '__main__':
    data = DstSeqData('Hello world', file='blah/blah.txt')
    print(data.file)









