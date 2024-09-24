
import ezpyz as ez
import dst.data.dst_data as dst

totals = {
    'DaringWookiee': 0,
    'FearlessBodhi': 0,
    'GPTpipeline': 0
}

for model in totals:
    data = ez.File(f'data/dsg_predictions/error_analysis/{model}/gptdst5k_100_test.pkl').load()
    print(model)
    for turn in data:
        print('   ', turn.turn)
        for slot, value in turn.predicted_slots.items():
            if value is not None:
                print(f"        {slot.name}: {', '.join(value)}")
                totals[model] += 1

print(totals)

errors = ez.File('src/dst/analysis/FinalErrorAnalysis.csv').load()

for row, total in zip(errors[1:], totals.values()):
    for i in range(1, len(row)):
        row[i] = float(row[i]) / total

ez.File('src/dst/analysis/FinalErrorAnalysisPercents.csv').save(errors)