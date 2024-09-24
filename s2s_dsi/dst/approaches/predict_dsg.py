
from dst.approaches.dst_seq_data import DstSeqData
from dst.approaches.seq2seq_dsg import Seq2seqDSG, NoException
from dst.approaches.t5 import T5
import ezpyz as ez
import traceback

def predict(
    data:str,
    model: str,
    load_in_8bit=False,
    predict_batch_size=2,
    max_new_tokens=256,
    gen_beams=1,
    gen_sampling=False,
    repetition_penalty:float=1.1,
    saving=True,
    email_notifications:str='jamesfinch293@gmail.com',
    catch_and_alert_errors=True,
    examples_to_predict_limit:int=None,
    predictions_path:str=None
):
    try:
        model = Seq2seqDSG.load(model, load_in_8bit=load_in_8bit)
        t5: T5 = model.seq2seq_model  # noqa
        t5.predict_batch_size = predict_batch_size
        t5.generation_config.max_new_tokens = max_new_tokens
        t5.generation_config.num_beams = gen_beams
        t5.generation_config.do_sample = gen_sampling
        t5.generation_config.repetition_penalty = repetition_penalty
    except (Exception if catch_and_alert_errors else NoException) as e:
        if email_notifications and '@' in email_notifications:
            ez.email(
                email_notifications,
                f'DSG {model.experiment} load error',
                traceback.format_exc()
            )
        traceback.print_exc()
        return
    datas = data.split(', ')
    for data in datas:
        data = DstSeqData.load(data)
        if examples_to_predict_limit is not None:
            data.dialogues = data.dialogues[:examples_to_predict_limit]
        try:
            predictions = model.predict(data)
        except (Exception if catch_and_alert_errors else NoException) as e:
            if email_notifications and '@' in email_notifications:
                ez.email(
                    email_notifications,
                    f'DSG {model.experiment} prediction error',
                    traceback.format_exc()
                )
            traceback.print_exc()
            print('\n\n', f"{model.hyperparameters.display()}")
            return
        if saving:
            predictions.save(predictions_path)
        if email_notifications and '@' in email_notifications:
            ez.email(
                email_notifications,
                f'DSG {model.experiment} predictions on {data.file.path.name[:4]} done',
                f"{model.hyperparameters.display()}"
            )

if __name__ == '__main__':

    import fire

    fire.Fire(predict)

    # predict(
    #     data='data/sgd-data/sgd_100_test.pkl',
    #     model='ex/Seq2seqDSG/InfiniteIridonia/1/model',
    #     load_in_8bit=True,
    #     predict_batch_size=1,
    #     max_new_tokens=256,
    #     gen_beams=5,
    #     gen_sampling=False,
    #     repetition_penalty=2.0,
    #     saving=True,
    #     examples_to_predict_limit=1,
    #     email_notifications='False',
    # )