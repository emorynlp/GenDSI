
instructions = """Your objective is to evaluate the quality of the dialogue state update for the last dialogue turn.

The dialogue state update is organized into slots (types of information) and values (specific instances of these types of information), like so:

    flight destination:
        New York City
    flight date:
        January 4
    flight available:
        yes

To evaluate dialogue state updates, direct your attention to the dialogue history in the middle pane, focusing on the most recent dialogue turn and corresponding dialogue state update.

The Task pane on the bottom left will prompt you with a yes/no question about the most recent turn: answer this prompt with a YES/ACCEPT answer by pressing [a] on your keyboard, or answer with NO/REJECT by pressing [r] on your keyboard.

After each decision, the next task will be automatically presented. If you wish to revisit a previous task, simply press [b]/[left arrow] on your keyboard (and [n]/[right arrow] to move forward again).

    [a]: affirm
    [r]: reject
    [b]/[leftarrow]: previous task
    [n]/[rightarrow]: next task

On the rightmost pane, you'll find a dynamic summary of your previous decisions, and the top [PROGRESS] summary shows which dialogue, turn, and task you're currently on.
"""

slot_is_correct_q = """Given a dialogue turn, a correct slot-value pair in the dialogue state update is one where:

* the slot name represents a type of information that is relevant to the current turn
* the value represents information mentioned or strongly implied by the turn (values representing information that is ONLY mentioned or implied in PREVIOUS turns are not correct)
* if the value is "?", the slot information type was requested (implicitly or explicitly) by the turn's speaker

Is the slot and value shown below an accurate extraction of information shared in the last dialogue turn?"""
slot_follows_specification_q = "Are the slot and value shown below valid; namely, the slot details a type of information and the value provides a realization of that type of information?\n\nFor this question, it does not matter whether the slot-value is accurate with respect to the last turn."
turn_state_is_complete_q = """Given a dialogue turn, a complete dialogue state update is one that covers all key information shared in the turn. To decide whether the state update is complete:
    
    1. Identify all key information shared in the turn-- key information represents information that is necessary for the listener to understand in order for the dialogue to be successful.
    2. For each piece of key information, check whether the dialogue state update contains a slot-value pair that covers the key information.
    3. All key information must be covered by the slot-value pairs for the dialogue state update to be complete.
    
Note that if the dialogue state update contains ADDITIONAL information that is irrelevant or incorrect, it does not affect the completeness of the dialogue state update: in other words, a dialogue state update can be complete even if it contains redundant or incorrect information, as long as the key information is covered.

Another way of deciding whether a dialogue state update is complete is to decide whether there is any MISSING key information: if key info is missing in the state update, the turn is not complete; otherwise, the turn is complete (note: this means that, rarely, a turn may have no key information and is thus complete by default).

Does the Dialogue State Update cover all key information that is shared in the last dialogue turn?"""
turn_state_is_redundant_q = "Does the 'Turn State' contain any slots that are redundant (i.e. capture the same information)?"