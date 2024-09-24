import dataclasses
from dataclasses import dataclass, field
from typing import Union, List

@dataclass
class SlotValue:
    name: str
    value: str
    is_correct: Union[bool,None] = None
    follows_specification: Union[bool,None] = None

    def __hash__(self):
        return id(self)

    def to_dict(self):
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(data):
        return SlotValue(**data)

@dataclass
class Turn:
    text: str
    slots: List[Union[SlotValue,None]]
    state_is_complete: Union[bool,None] = None
    state_is_redundant: Union[bool,None] = None
    skip: bool = False

    def __iter__(self):
        if self.slots is not None:
            return iter(self.slots)
        return iter([])

    def __getitem__(self, idx):
        if self.slots is not None:
            return self.slots[idx]
        return None

    def to_dict(self):
        self_dict = dataclasses.asdict(self)
        if self.slots:
            slots = [s.to_dict() for s in self.slots]
            self_dict['slots'] = slots
        return self_dict

    @staticmethod
    def from_dict(data):
        if data['slots']:
            slots = [SlotValue.from_dict(s) for s in data['slots']]
            data['slots'] = slots
        return Turn(**data)

@dataclass
class Dialogue:
    turns: List[Turn] = field(default_factory=list)
    model: str = None
    index: int = None

    def __iter__(self):
        return iter(self.turns)

    def __getitem__(self, idx):
        return self.turns[idx]

    def __len__(self):
        return len(self.turns)

    def to_dict(self):
        turns = [t.to_dict() for t in self.turns]
        self_dict = dataclasses.asdict(self)
        self_dict['turns'] = turns
        return self_dict

    @staticmethod
    def from_dict(data):
        turns = [Turn.from_dict(t) for t in data['turns']]
        data['turns'] = turns
        return Dialogue(**data)

@dataclass
class Dialogues:
    collection: List[Dialogue] = field(default_factory=list)

    def __iter__(self):
        return iter(self.collection)

    def __getitem__(self, idx):
        return self.collection[idx]

    def __len__(self):
        return len(self.collection)

    def to_dict(self):
        dialogues = [d.to_dict() for d in self.collection]
        self_dict = dataclasses.asdict(self)
        self_dict['collection'] = dialogues
        return self_dict

    @staticmethod
    def from_dict(data):
        dialogues = [Dialogue.from_dict(d) for d in data['collection']]
        data['collection'] = dialogues
        return Dialogues(**data)

@dataclass
class SlotTask:
    obj: SlotValue
    question: str
    update_func: callable
    retrieve_func: callable
    identifier: tuple[str, str]

@dataclass
class TurnTask:
    obj: Turn
    question: str
    update_func: callable
    retrieve_func: callable
    identifier: tuple[str, str]


