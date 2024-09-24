
from promptium.prompt import prompt

@prompt(model='gpt-4')
def write_code():
    """
    Make a python GUI application, where the goal is to annotate a dialogue dataset. The dialogues are represented in this JSON format, where each dialogue turn is an object with a "text" field representing what was said:

    [
        [
            {"text": "Hello, how are you?"},
            {"text": "Good, you?"},
            {"text": "I'm fine, thanks."}
        ],
        [
            {"text": "Hey, good weather today."},
            {"text": "Yeah, it's really nice."},
            {"text": "Indeed."}
        ],
    ]

    And this should be how the JSON looks after annotating:

    [
        [
            {"text": "Hello, how are you?", "annotation": true},
            {"text": "Good, you?", "annotation": true},
            {"text": "I'm fine, thanks.", "annotation": false}
        ],
        [
            {"text": "Hey, good weather today.", "annotation": true},
            {"text": "Yeah, it's really nice.", "annotation": false},
            {"text": "Indeed.", "annotation": false}
        ],
    ]

    The user will do annotations using 4 keys:
    1. the "a" key that registers the turn as good, "annotation": true
    2. the "r" key that registers the turn as bad, "annotation": false
    3. the "n" key that skips forward one turn
    4. the "b" key that goes backward one turn

    Make sure the following requirements are also met:
    * When annotating a turn, all previous turns in the dialogue are displayed
    * When pressing the "a" or "r" keys, the app automatically takes the user to the next turn
    * The application saves the dialogues JSON with the annotations each time an annotation is made
    * The application skips forward to the first un-annotated turn when opened
    """

import json
import tkinter as tk
from tkinter import font as tkfont
from tkinter.scrolledtext import ScrolledText

class App:
    def __init__(self, root, dialogues_path):
        self.dialogues_file = dialogues_path
        with open(self.dialogues_file) as file:
            self.dialogues_data = json.load(file)
        self.font = tkfont.Font(family="Courier", size=16)
        self.curr_dialogue = 0
        self.curr_turn = 0
        # skip to first un-annotated turn
        while 'annotation' in self.dialogues_data[self.curr_dialogue][self.curr_turn]:
            self.next_turn()
            if self.curr_dialogue == 0 and self.curr_turn == 0:
                break
        self.dialogue_window = ScrolledText(root, font=self.font, width=120, height=50)
        self.dialogue_window.pack()
        # bind keys
        root.bind('a', self.annotate_good)
        root.bind('r', self.annotate_bad)
        root.bind('n', self.to_next_turn)
        root.bind('b', self.to_previous_turn)
        self.update_display()

    def update_display(self):
        self.dialogue_window.delete('1.0', tk.END)
        for i, turn in enumerate(self.dialogues_data[self.curr_dialogue]):
            line = turn['text']
            self.dialogue_window.insert(tk.END, line + '\n')
            if i == self.curr_turn:
                break

    def annotate(self, is_good):
        self.dialogues_data[self.curr_dialogue][self.curr_turn]['annotation'] = is_good
        with open(self.dialogues_file, 'w') as file:
            json.dump(self.dialogues_data, file)
        self.next_turn()

    def annotate_good(self, event=None):
        self.annotate(True)

    def annotate_bad(self, event=None):
        self.annotate(False)

    def next_turn(self):
        self.curr_turn += 1
        if self.curr_turn >= len(self.dialogues_data[self.curr_dialogue]):
            self.curr_turn = 0
            self.curr_dialogue += 1
            if self.curr_dialogue >= len(self.dialogues_data):
                self.curr_dialogue = 0
        self.update_display()

    def to_next_turn(self, event=None):
        self.next_turn()

    def to_previous_turn(self, event=None):
        self.curr_turn -= 1
        if self.curr_turn < 0:
            self.curr_dialogue -= 1
            if self.curr_dialogue < 0:
                self.curr_dialogue = len(self.dialogues_data) - 1
            self.curr_turn = len(self.dialogues_data[self.curr_dialogue]) - 1
        self.update_display()


def main():
    root = tk.Tk()
    root.geometry("1500x500")
    app = App(root, 'data/scratch/dialogues.json')
    root.mainloop()


import ezpyz as ez
dialogues = [
    [
        {"text": "Hello, how are you?"},
        {"text": "Good, you?"},
        {"text": "I'm fine, thanks."},
        {"text": "Well nobody cares."}
    ],
    [
        {"text": "Hey, good weather today."},
        {"text": "Yeah, it's really nice."},
        {"text": "Indeed."},
        {"text": "I'm going to go now."}
    ]
]
ez.File('data/scratch/dialogues.json').save(dialogues)


if __name__ == "__main__":
    main()

