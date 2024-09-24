
import dst.data.dst_data as dst


data = dst.DstData.load("data/sgd-data/sgd_full.pkl")
turns = [turn for dialogue in data.dialogues for turn in dialogue.turns]
turn_slot_values = [
    (turn, slot, value) for turn in turns if turn.slots is not None
    for slot, value in turn.slots.items()
]
turn_domains = {}
for turn in turns:
    turn_domains[turn] = {slot.domain for slot in turn.slots} if turn.slots else None
multi_domain_turns = {turn: domains for turn, domains in turn_domains.items() if domains and len(domains) > 1}
domains = {}
for turn, slot, values in turn_slot_values:
    domain = slot.domain
    domains.setdefault(domain, set()).add(turn)
...




if __name__ == '__main__':
    ...