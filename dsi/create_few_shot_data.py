
import ezpyzy as ez
import random as rng
import dsi.dst_data as dst
import dsi.cluster as dcl


def create_few_shot_data(data_path, induction_output_path):
    data = dst.Data(data_path=data_path)
    induced = dcl.Clustered.of(induction_output_path)
    turn_id_to_dialogue = dict(zip(data.turns.turn_id, data.turns.dialogue))
    turn_id_to_domain = dict(zip(data.turns.turn_id, data.turns.domain))
    svid_to_value = dict(zip(data.slot_values.slot_value_id, data.slot_values.value))
    cluster_domain_dialogue_svids = {}
    for turn_id, svid, cluster in zip(induced.turn_id, induced.slot_value_id, induced.cluster_id):
        if cluster == -1:
            continue
        elif svid_to_value[svid] == '?':
            continue
        domain = turn_id_to_domain[turn_id]
        dialogue = turn_id_to_dialogue[turn_id]
        if cluster not in cluster_domain_dialogue_svids:
            cluster_domain_dialogue_svids[cluster] = {}
        if domain not in cluster_domain_dialogue_svids[cluster]:
            cluster_domain_dialogue_svids[cluster][domain] = {}
        if dialogue not in cluster_domain_dialogue_svids[cluster][domain]:
            cluster_domain_dialogue_svids[cluster][domain][dialogue] = []
        cluster_domain_dialogue_svids[cluster][domain][dialogue].append(svid)
    clustered_svids = set()
    svid_alts = [{}, {}, {}, {}] # n -> svid -> alts
    for cluster, domain_dialogue_svids in cluster_domain_dialogue_svids.items():
        for domain, dialogue_svids in domain_dialogue_svids.items():
            svids_in_cluster_domain = {
                svid for dialogue, svids in dialogue_svids.items() for svid in svids}
            for dialogue, svids in dialogue_svids.items():
                dialogue_svids = set(svids)
                clustered_svids.update(dialogue_svids)
                alts = svids_in_cluster_domain - dialogue_svids
                for svid in svids:
                    sas = [
                        a for a in alts if svid_to_value[a] not in {'?', svid_to_value[svid]}]
                    if len(sas) >= 3:
                        svid_alts[3][svid] = sas
                    else:
                        svid_alts[len(sas)][svid] = sas
    unclustered_svids = set(data.slot_values.slot_value_id) - clustered_svids
    svid_alts[0].update({svid: [] for svid in unclustered_svids})
    num_total_svids = len(data.slot_values.slot_value_id)
    num_quarter_svids = num_total_svids // 4
    print(f"Number of total svids: {num_total_svids}")
    for i in range(3, 0, -1):
        if len(svid_alts[i]) > num_quarter_svids:
            svid_list = list(svid_alts[i].items())
            rng.shuffle(svid_list)
            svid_alts[i] = dict(svid_list[:num_quarter_svids])
            svid_alts[i-1].update(svid_list[num_quarter_svids:])
        print(f'Number of {i} shots: {len(svid_alts[i])}')
    svid_to_shots = {}
    for n in range(1, 4):
        n_svid_alts = svid_alts[n]
        for svid, alts in n_svid_alts.items():
            svid_shots = rng.sample(alts, n)
            svid_to_shots[svid] = svid_shots
    svid_to_turns = dict(zip(data.slot_values.slot_value_id, data.slot_values.turn_id))
    svid_to_slot = dict(zip(data.slot_values.slot_value_id, data.slot_values.slot))
    svid_to_sid = dict(zip(data.slot_values.slot_value_id, data.slot_values.slot_id))
    sid_to_description = dict(zip(data.slots.slot_id, data.slots.description))
    turn_to_text = dict(zip(data.turns.turn_id, data.turns.text))
    new_descriptions = {}
    for svid in data.slot_values.slot_value_id:
        description = sid_to_description[svid_to_sid[svid]]
        if svid not in svid_to_shots:
            continue
        exs = ''.join(
            f"\n  ex. {turn_to_text[svid_to_turns[shot]]} -> {svid_to_slot[svid]}? {svid_to_value[shot]}"
            for shot in svid_to_shots[svid])
        if rng.random() < 0.5:
            desc = ''
        else:
            desc = description
        new_description = f"{desc}{exs}\n"
        if rng.random() < 0.01:
            print(f"New description:\n{new_description}\nwith value {svid_to_value[svid]}\n\n")
        new_descriptions[svid_to_sid[svid]] = new_description
    data.slots.description[:] = [
        new_descriptions.get(s, d) for s, d in list(zip(data.slots.slot_id, data.slots.description))]
    print("\n\n".join(data.slots.description[:30]))
    print("num slots:", len(data.slots))
    print(data.slots[:5])
    data.save('data/dsg5k/fewshot')




if __name__ == '__main__':
    create_few_shot_data('data/dsg5k/train', 'data/dsg5k/dsg5k_induced.csv')
