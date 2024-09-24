
import os, json
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
from tkinter.scrolledtext import ScrolledText
from data_spec import Dialogues, SlotTask, TurnTask, Union, List
from instructions import *
from idlelib.tooltip import Hovertip

###############################################################
##
## Annotation Task Specifications
##
###############################################################

set_slot_is_correct = lambda slotvalue_obj, decision: setattr(slotvalue_obj, 'is_correct', decision)
set_slot_follows_specification = lambda slotvalue_obj, decision: setattr(slotvalue_obj, 'follows_specification', decision)
set_turn_state_is_complete = lambda turn_obj, decision: setattr(turn_obj, 'state_is_complete', decision)
set_turn_state_is_redundant = lambda turn_obj, decision: setattr(turn_obj, 'state_is_redundant', decision)

get_slot_is_correct = lambda slotvalue_obj: getattr(slotvalue_obj, 'is_correct')
get_slot_follows_specification = lambda slotvalue_obj: getattr(slotvalue_obj, 'follows_specification')
get_turn_state_is_complete = lambda turn_obj: getattr(turn_obj, 'state_is_complete')
get_turn_state_is_redundant = lambda turn_obj: getattr(turn_obj, 'state_is_redundant')

SLOT_IS_CORRECT = 'slot_is_correct'
SLOT_FOLLOWS_SPECIFICATION = 'slot_follows_specification'
TURN_STATE_IS_COMPLETE = 'turn_state_is_complete'
TURN_STATE_IS_REDUNDANT = 'turn_state_is_redundant'

TASKS_SETUP = {
    SLOT_IS_CORRECT: (slot_is_correct_q, set_slot_is_correct, get_slot_is_correct, ('slot', 'Correct')),
    SLOT_FOLLOWS_SPECIFICATION: (slot_follows_specification_q, set_slot_follows_specification, get_slot_follows_specification, ('slot', 'Valid')),
    TURN_STATE_IS_COMPLETE: (turn_state_is_complete_q, set_turn_state_is_complete, get_turn_state_is_complete, ('turn', 'Complete')),
    TURN_STATE_IS_REDUNDANT: (turn_state_is_redundant_q, set_turn_state_is_redundant, get_turn_state_is_redundant, ('turn', 'Redundant')),
}

###############################################################
##
## Annotation Tasks To Include
##
###############################################################

INCLUDE_TASKS = [
    SLOT_IS_CORRECT,
    # SLOT_FOLLOWS_SPECIFICATION,
    TURN_STATE_IS_COMPLETE,
    # TURN_STATE_IS_REDUNDANT,
]

###############################################################
##
## Tooltip Over Table
##
###############################################################

def split_text_into_chunks(text, chunk_size=60):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

class ItemToolTip(Hovertip):
    # https://comp.lang.tcl.narkive.com/LjvITRzh/tooltips-for-ttk-treeview

    def __init__(self, widget):
        Hovertip.__init__(self, widget, text='', hover_delay=500)
        self._id4 = widget.bind("<Motion>", self.update_text)
        self.last_focus = None

    def update_text(self, event=None):
        self.unschedule()
        self.hidetip()
        widget = event.widget
        _iid = widget.identify_row(event.y)
        text = ' || '.join([str(x) for x in widget.item(_iid)['values']])
        self.text = '\n'.join(split_text_into_chunks(text, chunk_size=60))
        if _iid != self.last_focus:
            if self.last_focus and widget.exists(self.last_focus):
                widget.item(self.last_focus, tags=[])
            widget.item(_iid, tags=['focus'])
            self.last_focus = _iid
        if self.text != '':
            self.schedule()


###############################################################
##
## Interface
##
###############################################################

class App:
    def __init__(self, root, dialogues_path):
        self.dialogues_file = dialogues_path
        with open(self.dialogues_file) as file:
            raw_data = json.load(file)
            self.dialogues_data: Dialogues = Dialogues.from_dict(raw_data)
        self.font = tkfont.Font(family="Courier", size=20)
        self.smaller_font = tkfont.Font(family="Courier", size=18)
        self.text_font = tkfont.Font(family="Arial", size=20)
        self.bold_text_font = tkfont.Font(family="Arial", size=20, weight="bold")
        self.results_font = tkfont.Font(family="Courier", size=20)
        self.curr_dialogue = 0
        self.curr_turn = 0
        self.curr_annotation = 0
        self.tasks: List[SlotTask|TurnTask] = self.get_turn_tasks()

        # skip to first unfinished annotation task
        while True:
            break_inner = False
            for i, t in enumerate(self.tasks):
                if t.retrieve_func(t.obj) is None:
                    self.curr_annotation = i
                    break_inner = True
                    break
            else:
                get_next = True
                if self.curr_dialogue == len(self.dialogues_data) - 1:
                    # special case for when on last annotation task of the set
                    get_next = False
                    for turn in self.dialogues_data[self.curr_dialogue].turns[self.curr_turn + 1:]:
                        if not turn.skip:
                            get_next = True
                            break
                if get_next:
                    self.next_turn()
                else:
                    self.curr_annotation = len(self.tasks) - 1
                    break
            if break_inner:
                break

        def calculate_width(percent):
            return int(root.winfo_width() * (percent / 100))

        def calculate_height(percent):
            return int(root.winfo_height() * (percent / 100))

        col = [35, 45, 20]
        row = [20, 2, 43, 35]
        root.columnconfigure(0, weight=col[0])
        root.columnconfigure(1, weight=col[1])
        root.columnconfigure(2, weight=col[2])
        root.rowconfigure(0, weight=row[0], minsize=30)
        root.rowconfigure(1, weight=row[1], minsize=30)
        root.rowconfigure(2, weight=row[2])
        root.rowconfigure(3, weight=row[3])

        padx = 5
        pady = 5


        self.progress_window = tk.Label(
            root, font=self.smaller_font, width=calculate_width(sum(col[0:])), height=calculate_height(row[0]), bg='gray80'
        )
        self.progress_window.grid(row=0, column=0, columnspan=3, padx=padx, pady=pady, sticky='nsew')




        self.instruction_label = tk.Label(
            root, font=self.bold_text_font, text="Instructions", width=calculate_width(col[0]), height=calculate_height(row[1])
        )
        self.instruction_label.grid(row=1, column=0, padx=padx, pady=pady, sticky='nsew')

        self.dialogue_label = tk.Label(
            root, font=self.bold_text_font, text="Dialogue History", width=calculate_width(col[1]), height=calculate_height(row[1])
        )
        self.dialogue_label.grid(row=1, column=1, padx=padx, pady=pady, sticky='nsew')

        self.turn_state_label = tk.Label(
            root, font=self.bold_text_font, text="Dialogue State Update", width=calculate_width(col[2]), height=calculate_height(row[1])
        )
        self.turn_state_label.grid(row=1, column=2, padx=padx, pady=pady, sticky='nsew')




        self.instruction_window = ScrolledText(
            root, font=self.text_font, width=calculate_width(col[0]), height=calculate_height(row[2]), wrap=tk.WORD
        )
        self.instruction_window.grid(row=2, column=0, padx=padx, pady=pady, sticky='nsew')

        self.dialogue_window = ScrolledText(
            root, font=self.font, width=calculate_width(col[1]), height=calculate_height(sum(row[2:])), wrap=tk.WORD
        )
        self.dialogue_window.grid(row=2, column=1, rowspan=2, padx=padx, pady=pady, sticky='nsew')
        self.dialogue_window.tag_configure("S1", foreground="black")
        self.dialogue_window.tag_configure("slotvaluesS1", foreground="green", font=self.smaller_font)
        self.dialogue_window.tag_configure("S2", foreground="gray40")
        self.dialogue_window.tag_configure("slotvaluesS2", foreground="green", font=self.smaller_font)

        style = ttk.Style()
        style.configure("Treeview", font=tkfont.Font(family="Courier", size=16))
        self.turn_state_table_columns = ['Slot', 'Value'] + [TASKS_SETUP[t][-1][1] for t in INCLUDE_TASKS if TASKS_SETUP[t][-1][0] == 'slot']
        self.turn_state_table = ttk.Treeview(
            root, columns=self.turn_state_table_columns, show='headings', height=calculate_height(row[2])
        )
        self.turn_state_table.tag_configure('focus', background='yellow')
        self.turn_state_table.grid(row=2, column=2, padx=padx, pady=pady, sticky='nsew')
        for i, c in enumerate(self.turn_state_table_columns):
            self.turn_state_table.heading(c, text=c)
            self.turn_state_table.column(c, width=calculate_width(col[2]) // len(self.turn_state_table_columns), anchor="center")
        self.turn_table_tooltip = ItemToolTip(
            self.turn_state_table
        )






        self.annotation_prompt = ScrolledText(
            root, font=self.text_font, width=calculate_width(col[0]), height=calculate_height(row[3]), wrap=tk.WORD
        )
        self.annotation_prompt.grid(row=3, column=0, padx=padx, pady=pady, sticky='nsew')
        self.annotation_prompt.tag_configure("bold_centered_text", font=self.bold_text_font, justify="center")


        self.turn_state_annotations = ScrolledText(
            root, font=tkfont.Font(family="Courier", size=16), width=calculate_width(col[2]), height=calculate_height(row[3]), wrap=tk.WORD
        )
        self.turn_state_annotations.grid(row=3, column=2, padx=padx, pady=pady, sticky='nsew')
        self.turn_state_annotations.tag_configure("bold_centered_text", font=self.bold_text_font, justify="center")




        root.bind('a', self.accept)
        root.bind('r', self.reject)
        root.bind('n', self.to_next)
        root.bind('b', self.to_previous)
        root.bind("<Right>", self.to_next)
        root.bind("<Left>", self.to_previous)
        self.update_display()

    def get_turn_tasks(self) -> List[Union[SlotTask,TurnTask]]:
        tasks = []
        curr_turn = self.dialogues_data[self.curr_dialogue][self.curr_turn]
        if curr_turn.skip:
            return tasks
        for slotvalue_obj in curr_turn:
            if SLOT_IS_CORRECT in INCLUDE_TASKS:
                tasks.append(SlotTask(slotvalue_obj, *TASKS_SETUP[SLOT_IS_CORRECT]))
            if SLOT_FOLLOWS_SPECIFICATION in INCLUDE_TASKS:
                tasks.append(SlotTask(slotvalue_obj, *TASKS_SETUP[SLOT_FOLLOWS_SPECIFICATION]))
        if TURN_STATE_IS_COMPLETE in INCLUDE_TASKS:
            tasks.append(TurnTask(curr_turn, *TASKS_SETUP[TURN_STATE_IS_COMPLETE]))
        if TURN_STATE_IS_REDUNDANT in INCLUDE_TASKS:
            tasks.append(TurnTask(curr_turn, *TASKS_SETUP[TURN_STATE_IS_REDUNDANT]))
        return tasks

    def update_display(self):
        self.dialogue_window.configure(state='normal')
        self.dialogue_window.delete('1.0', tk.END)
        for i, turn in enumerate(self.dialogues_data[self.curr_dialogue]):
            line = turn.text
            self.dialogue_window.insert(tk.END, line + '\n\n', "S1" if i % 2 == 0 else "S2")
            if i == self.curr_turn:
                break

        self.annotation_prompt.configure(state='normal')
        self.annotation_prompt.delete('1.0', tk.END)

        curr_task = self.tasks[self.curr_annotation] if self.curr_annotation > -1 else None
        curr_tag = 'S1' if self.curr_turn % 2 == 0 else 'S2'
        if isinstance(curr_task, SlotTask):
            self.annotation_prompt.insert(tk.END, f'Task: Slot-Value {curr_task.identifier[1]}' + '\n\n', 'bold_centered_text')
            self.annotation_prompt.insert(tk.END, curr_task.question + '\n\n')
            self.annotation_prompt.insert(tk.END, f"    {curr_task.obj.name}:\n")
            self.annotation_prompt.insert(tk.END, f"        {curr_task.obj.value}")
            self.dialogue_window.insert(tk.END, f"    {curr_task.obj.name}:\n", f"slotvalues{curr_tag}")
            self.dialogue_window.insert(tk.END, f"        {curr_task.obj.value}", f"slotvalues{curr_tag}")
            self.dialogue_window.insert(tk.END, f'\n\n    {"-"*15}\n\n    {curr_task.identifier[1]}?', f"slotvalues{curr_tag}")
        elif isinstance(curr_task, TurnTask):
            self.annotation_prompt.insert(tk.END, f'Task: Dialogue State Update {curr_task.identifier[1]}' + '\n\n', 'bold_centered_text')
            self.annotation_prompt.insert(tk.END, curr_task.question)
            if not curr_task.obj.skip:
                if curr_task.obj.slots:
                    for i, slotvalue_obj in enumerate(curr_task.obj.slots):
                        self.dialogue_window.insert(tk.END, f"    {slotvalue_obj.name}:\n", f"slotvalues{curr_tag}")
                        if i < len(curr_task.obj.slots) - 1:
                            self.dialogue_window.insert(tk.END, f"        {slotvalue_obj.value}\n\n", f"slotvalues{curr_tag}")
                        else:
                            self.dialogue_window.insert(tk.END, f"        {slotvalue_obj.value}", f"slotvalues{curr_tag}")
                else:
                    self.dialogue_window.insert(tk.END, f"    <no slots>", f"slotvalues{curr_tag}")
            self.dialogue_window.insert(tk.END, f'\n\n    {"-" * 15}\n\n    {curr_task.identifier[1]}?', f"slotvalues{curr_tag}")

        self.annotation_prompt.configure(state='disabled')
        self.dialogue_window.configure(state='disabled')

        self.turn_state_table.delete(*self.turn_state_table.get_children())
        self.turn_state_annotations.delete('1.0', tk.END)
        printed = set()
        for i, task in enumerate(self.tasks):
            if i == self.curr_annotation + 1:
                break
            if isinstance(task, SlotTask) and task.obj not in printed:
                values = (
                    task.obj.name,
                    task.obj.value,
                    *[str(TASKS_SETUP[t][2](task.obj)) for t in INCLUDE_TASKS if TASKS_SETUP[t][-1][0] == 'slot']
                )
                self.turn_state_table.insert(
                    '',
                    'end',
                    values=values,
                )
                printed.add(task.obj)
            if isinstance(task, TurnTask):
                if "Dialogue State Characteristics" not in self.turn_state_annotations.get("1.0", tk.END):
                    self.turn_state_annotations.insert(tk.END, f'Dialogue State Update Characteristics:\n\n', "bold_centered_text")
                self.turn_state_annotations.insert(tk.END, f'    {task.identifier[1]}: {task.retrieve_func(task.obj)}\n\n')

        self.instruction_window.configure(state='normal')
        self.instruction_window.delete('1.0', tk.END)
        self.instruction_window.insert(tk.END, instructions)
        self.instruction_window.configure(state='disabled')

        annotatable_turns = [t for t in self.dialogues_data[self.curr_dialogue] if not t.skip]
        annotatable_turn_idxs = [i for i,t in enumerate(self.dialogues_data[self.curr_dialogue]) if not t.skip]
        curr_annotatable_turn_idx = annotatable_turn_idxs.index(self.curr_turn) + 1
        self.progress_window.config(
            text=f"[PROGRESS]   Dialogue: {self.curr_dialogue+1} / {len(self.dialogues_data)}  ||  "
                 f"Turn: {curr_annotatable_turn_idx} / {len(annotatable_turns)}  ||  "
                 f"Task: {self.curr_annotation + 1} / {len(self.tasks)}  "
        )

        self.dialogue_window.see(tk.END)


    def annotate(self, decision):
        if self.curr_annotation < 0:
            return
        curr_task = self.tasks[self.curr_annotation]
        curr_task.update_func(curr_task.obj, decision)
        with open(self.dialogues_file, 'w') as file:
            data_to_dump = self.dialogues_data.to_dict()
            json.dump(data_to_dump, file, indent=2)
        self.next()

    def accept(self, event=None):
        self.annotate(True)

    def reject(self, event=None):
        self.annotate(False)

    def next(self):
        # special case for when on last annotation task of the set
        if self.curr_dialogue == len(self.dialogues_data) - 1 and self.curr_annotation == len(self.tasks) - 1:
            for turn in self.dialogues_data[self.curr_dialogue].turns[self.curr_turn+1:]:
                if not turn.skip:
                    break # next turn has annotation task
            else:
                return # no next turns have annotation tasks
        self.curr_annotation += 1
        while self.curr_annotation >= len(self.tasks):
            self.next_turn()
        self.update_display()

    def next_turn(self):
        self.curr_annotation = 0
        self.curr_turn += 1
        if self.curr_turn >= len(self.dialogues_data[self.curr_dialogue]):
            self.curr_turn = 0
            self.curr_dialogue += 1
            if self.curr_dialogue >= len(self.dialogues_data):
                # do not loop back to beginning; just end on last dialogue, turn, and annotation task
                self.curr_dialogue -= 1
                self.curr_turn = len(self.dialogues_data[self.curr_dialogue]) - 1
                self.curr_annotation = len(self.tasks) - 1
                return False
        self.tasks = self.get_turn_tasks()
        return True

    def to_next(self, event=None):
        if self.curr_annotation < 0:
            return
        elif self.tasks[self.curr_annotation].retrieve_func(self.tasks[self.curr_annotation].obj) is not None:
            self.next()
        else:
            if "(a)ccept or (r)eject" not in self.dialogue_window.get("1.0", tk.END):
                self.dialogue_window.configure(state='normal')
                self.dialogue_window.insert(tk.END, '\n\nYou cannot continue until you either (a)ccept or (r)eject!', "warning")
                self.dialogue_window.tag_configure("warning", foreground="red")
                self.dialogue_window.configure(state='disabled')
                self.dialogue_window.see(tk.END)

    def previous_turn(self):
        self.curr_turn -= 1
        if self.curr_turn < 0:
            self.curr_dialogue -= 1
            if self.curr_dialogue < 0:
                self.curr_dialogue = 0
                self.curr_turn = 0
                self.curr_annotation = 0
                return
            self.curr_turn = len(self.dialogues_data[self.curr_dialogue]) - 1
        self.tasks = self.get_turn_tasks()
        self.curr_annotation = len(self.tasks) - 1

    def to_previous(self, event=None):
        # special case for when on first annotation task of the set
        if self.curr_dialogue == 0 and self.curr_annotation == 0:
            for turn in self.dialogues_data[self.curr_dialogue].turns[:self.curr_turn]:
                if not turn.skip:
                    break # previous turn has annotation task
            else:
                return # no previous turns have annotation tasks
        self.curr_annotation -= 1
        while self.curr_annotation < 0:
            self.previous_turn()
        self.update_display()


###############################################################
##
## Main App
##
###############################################################

def main(dialogues_filename):
    root = tk.Tk()
    root.geometry("1500x500")
    app = App(root, dialogues_filename)
    root.mainloop()

if __name__ == "__main__":
    dialogues_filename = [
        f"annotation/{f}"
        for f in os.listdir('annotation/')
        if f.endswith('.json')
    ][0]
    main(dialogues_filename)

