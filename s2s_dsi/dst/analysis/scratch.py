
import ezpyz as ez
import pandas as pd


def something_error_analysis():

    errors = ez.File('src/dst/analysis/FinalErrorAnalysis.csv').load()
    errordf = pd.DataFrame.from_records(errors[1:], columns=errors[0])
    errordf.rename({'': 'Model'}, axis=1, inplace=True)


    for model in ('DaringWookiee', 'FearlessBodhi', 'GPTpipeline'):
        print('\n', '=' * 30 + f" {model} " + '=' * 30, '\n')
        error_analysis_data = ez.File(
            f'/home/jdfinch/Downloads/dsg_predictions/error_analysis/{model}/gptdst5k_100_test.pkl'
        ).load()
        slots = sum(
            len(
                [z for z, y in x.predicted_slots.items() if y is not None]
            ) for x in error_analysis_data
        )


from dst.utils import download

if __name__ == '__main__':
    download('h100', 'data/sgd_wo_domains/valid_DSG')


