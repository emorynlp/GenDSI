
import ezpyzy as ez

results = ez.File('results/induction_mwoz_valid.csv').load()
header = results[0]
results = results[1:]
fixed_results = []
for row in results: # noqa
    row = dict(zip(header, row))
    if row['precision'] == ')':
        del row['precision']
    fixed = list(row.values())
    fixed_results.append(fixed)

results = ez.File('results/induction_mwoz_valid.csv')
results.save([header] + fixed_results)