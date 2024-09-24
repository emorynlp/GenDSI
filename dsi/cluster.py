
import dataclasses as dc
import random

import ezpyzy as ez


@dc.dataclass
class ClusteredPoint(ez.Table):
    slot_value_id: ez.ColStr = None
    slot: ez.ColStr = None
    value: ez.ColStr = None
    domain: ez.ColStr = None
    cluster_id: ez.ColStr = None
    turn_id: ez.ColStr = None
    point_encoding: ez.ColObj = None

    def samples(self, silver, gold=None, matches=None, n=None):
        clusters = {}
        points = list(zip(self.cluster_id, self.slot_value_id, self.turn_id))
        for cluster_id, slot_value_id, turn_id in points:
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append((slot_value_id, turn_id))
        if n:
            for cluster_id, cluster_points in clusters.items():
                clusters[cluster_id] = random.sample(cluster_points, min(n, len(cluster_points)))
        dialogue_index_to_turn = {}
        for dialogue_id, turn_index, turn_id in zip(
            silver.turns.dialogue, silver.turns.turn_index, silver.turns.turn_id
        ):
            dialogue_index_to_turn[dialogue_id, turn_index] = turn_id
        turn_id_to_text = {}
        for turn_id, speaker, text in zip(
            silver.turns.turn_id, silver.turns.speaker, silver.turns.text
        ):
            turn_id_to_text[turn_id] = text
        turn_id_to_context = {}
        for dialogue_id, turn_index, turn_id in zip(
            silver.turns.dialogue, silver.turns.turn_index, silver.turns.turn_id
        ):
            context = []
            for i in range(1, turn_index+1):
                context.append(turn_id_to_text[dialogue_index_to_turn[dialogue_id, i]])
            turn_id_to_context[turn_id] = context
        turn_id_to_gold_slot_values = {}
        if gold:
            for turn_id, slot, value in zip(gold.turns.turn_id, gold.slot_values.slot, gold.slot_values.value):
                if turn_id not in turn_id_to_gold_slot_values:
                    turn_id_to_gold_slot_values[turn_id] = {}
                turn_id_to_gold_slot_values[turn_id][slot] = value
        turn_id_to_silver_slot_values = {}
        for turn_id, slot, value in zip(silver.slot_values.turn_id, silver.slot_values.slot, silver.slot_values.value):
            if turn_id not in turn_id_to_silver_slot_values:
                turn_id_to_silver_slot_values[turn_id] = {}
            turn_id_to_silver_slot_values[turn_id][slot] = value
        silver_svids_to_slot_value = {}
        for svid, slot, value in zip(silver.slot_values.slot_value_id, silver.slot_values.slot, silver.slot_values.value):
            silver_svids_to_slot_value[svid] = (slot, value)
        turn_id_to_dialogue_index = {}
        for dialogue_id, turn_index, turn_id in zip(
            silver.turns.dialogue, silver.turns.turn_index, silver.turns.turn_id
        ):
            turn_id_to_dialogue_index[turn_id] = (dialogue_id, turn_index)
        points = []
        for cluster_id, cluster_points in clusters.items(): # noqa
            for slot_value_id, turn_id in cluster_points:
                dialogue_id, turn_index = turn_id_to_dialogue_index[turn_id]
                previous_turn_id = dialogue_index_to_turn.get((dialogue_id, turn_index-1), None)
                if previous_turn_id:
                    previous_silver_slot_values = turn_id_to_silver_slot_values.get(previous_turn_id, {})
                else:
                    previous_silver_slot_values = {}
                text = turn_id_to_text[turn_id]
                context = '\n-  '.join(turn_id_to_context[turn_id])
                gold_slot_values = turn_id_to_gold_slot_values.get(turn_id, {})
                silver_slot_values = turn_id_to_silver_slot_values.get(turn_id, {})
                silver_slot_values.update(previous_silver_slot_values)
                slot, value = silver_svids_to_slot_value[slot_value_id]
                if matches:
                    matched_slot = matches.get(cluster_id, None)
                else:
                    matched_slot = None
                points.append(dict(
                    turn_id=turn_id,
                    slot_value_id=slot_value_id,
                    cluster_id=cluster_id,
                    matched_slot=matched_slot,
                    text=text,
                    context=context,
                    slot=slot,
                    value=value,
                    gold_slot_values=gold_slot_values,
                    silver_slot_values=silver_slot_values,
                ))
        points = Clustered.of(points)
        return points


@dc.dataclass
class Clustered(ez.Table):
    turn_id: ez.ColStr = None
    slot_value_id: ez.ColStr = None
    cluster_id: ez.ColStr = None
    matched_slot: ez.ColStr = None
    text: ez.ColStr = None
    context: ez.ColStr = None
    slot: ez.ColStr = None
    value: ez.ColStr = None
    gold_slot_values: ez.ColObj = None
    silver_slot_values: ez.ColObj = None


@dc.dataclass
class Cluster(ez.Table):
    cluster_id: ez.ColID = None
    cluster_label: ez.ColStr = None
    cluster_encoding: ez.ColObj = None




