
from dst.approaches.t5 import T5, SequencesData

t5 = T5()
prediction = t5.predict(SequencesData(
    'Translate English to Spanish: How are you?'
))

for output in prediction:
    print(output.seq_output)

