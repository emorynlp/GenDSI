import pathlib
import copy

import torch
import torch as pt
import transformers.optimization
from torch.utils.data import DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    GenerationConfig
)
from datasets import Dataset

import contextlib
import tqdm
import typing as T
import json

import ezpyz as ez
from dst.approaches.sequences_data import SequencesData, Hyperparameters, SequenceResults

import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

class StopAfterOneEpoch(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        control.should_training_stop = True


class T5:
    load_in_8bit=False
    def __init__(self,
        directory='.',
        approach=None,
        experiment=None,
        checkpoint='t5-small',
        learning_rate=1e-4,
        weight_decay=0.0,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        epochs=1,
        max_length=512,
        max_new_tokens=16,
        predict_batch_size=1,
        gen_beams=1,
        gen_sampling=False,
        repetition_penalty=None,
        repetition_alpha=None,
        silence_training=True,
        load_in_8bit=False,
        device='cuda',
    ):
        self.directory = pathlib.Path(directory)
        self.approach = approach or type(self).__name__
        self.experiment = experiment or ez.denominate({
            experiment.name for experiment in
            ((self.directory/self.approach).iterdir() if (self.directory/self.approach).exists() else ())
        })
        self.iteration = 0
        self.hyperparameters = Hyperparameters(pathlib.Path(
            self.directory,self.approach,self.experiment,str(self.iteration),
            f"hyperparameters.pkl"
        ))
        self.checkpoint = checkpoint
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epochs = epochs
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.predict_batch_size = predict_batch_size
        self.gen_beams = gen_beams
        self.gen_sampling = gen_sampling
        self.repetition_penalty = repetition_penalty
        self.repetition_alpha = repetition_alpha
        self.silence_training = silence_training
        self.load_in_8bit = load_in_8bit
        self.device = device
        self.tokenizer = T5TokenizerFast.from_pretrained('t5-base',
            load_in_8bit=self.load_in_8bit,
            device_map='auto',
            model_max_length = max_length,
            padding_side = 'left',
            truncation_side = 'left',
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            checkpoint,
            load_in_8bit=self.load_in_8bit,
            device_map='auto'
        )
        if not self.load_in_8bit:
            self.model = self.model.to(device)
        seq2seq_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            pad_to_multiple_of=8,
            max_length=max_length,
            return_tensors='pt',
        )
        oneseq_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors='pt'
        )
        def collator(batch):
            with ez.shush():
                if batch and 'labels' in batch[0] and batch[0]['labels'] is not None:
                    collated = seq2seq_collator(batch)
                    return collated
                else:
                    collated =  oneseq_collator(batch)
                    return collated
        self.collator = collator
        self.training_args = TrainingArguments(
            output_dir='.garbage',
            bf16=True if not self.load_in_8bit else False,
            evaluation_strategy='epoch',
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            num_train_epochs=epochs,
            per_device_train_batch_size=self.train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            per_device_eval_batch_size=self.predict_batch_size,
            save_strategy='no',
            save_total_limit=1,
            load_best_model_at_end=False,
            resume_from_checkpoint=None,
            optim='adafactor',
            no_cuda=device == 'cpu'
        )
        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            num_beams=gen_beams,
            num_return_sequences=1,
            do_sample=gen_sampling,
            early_stopping=True,
            repetition_penalty=repetition_penalty,
            penalty_alpha=repetition_alpha,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.collator,
            train_dataset=None,
            eval_dataset=None,
            callbacks=[StopAfterOneEpoch()],
        )
        self.record_hyperparameters('init')

    def save(self, path:ez.filelike=None):
        if path is None:
            path = pathlib.Path(self.directory,self.approach,self.experiment,str(self.iteration), 'model')
        path = ez.File(path).path
        self.trainer.args.output_dir = path
        self.trainer.save_model(path)
        self.trainer.save_state()  # noqa
        if self.trainer.optimizer:
            torch.save(self.trainer.optimizer.state_dict(), path / 'optimizer.pt')
        if self.trainer.lr_scheduler:
            torch.save(self.trainer.lr_scheduler.state_dict(), path / 'scheduler.pt')
        scopy = copy.copy(self)
        scopy.tokenizer = None
        scopy.model = None
        scopy.collator = None
        scopy.training_args = None
        scopy.generation_config = None
        scopy.trainer = None
        ez.Cache(path / 'Seq2seq.pkl').save(scopy)

    @classmethod
    def load(cls, path:ez.filelike, directory=None, device=None, load_in_8bit=None):
        path = ez.File(path).path
        t5 = ez.Cache(path / 'Seq2seq.pkl').load()
        t5.__init__(
            directory=t5.directory if directory is None else directory,
            approach=t5.approach,
            experiment=t5.experiment,
            checkpoint=path,
            learning_rate=t5.learning_rate,
            weight_decay=t5.weight_decay,
            train_batch_size=t5.train_batch_size,
            epochs=t5.epochs,
            max_length=t5.max_length,
            max_new_tokens=t5.max_new_tokens,
            predict_batch_size=t5.predict_batch_size,
            gen_beams=t5.gen_beams,
            gen_sampling=t5.gen_sampling,
            repetition_penalty=t5.repetition_penalty,
            repetition_alpha=t5.repetition_alpha,
            silence_training=t5.silence_training,
            device=t5.device if device is None else device,
            load_in_8bit=t5.load_in_8bit if load_in_8bit is None else load_in_8bit,
        )
        return t5

    def record_hyperparameters(self, stage_type, **kwargs):
        self.hyperparameters.record(self,
            **kwargs,
            stage_type=stage_type,
        )

    def preprocess(self, data:SequencesData):
        encoded = []
        for seqs in data:
            seq_in, seq_out = seqs.seq_input, seqs.seq_label
            self.tokenizer.truncation_side = 'left'
            encoded_seq_in = self.tokenizer(seq_in, truncation=True)
            self.tokenizer.truncation_side = 'right'
            encoded_seq_out = ez.option(self.tokenizer)(seq_out, truncation=True)
            encoded.append((encoded_seq_in, encoded_seq_out))
        return encoded

    def predict(self, data:SequencesData):
        encoded = self.preprocess(data)
        dataset = Dataset.from_list([
            {
                'input_ids': seq_in['input_ids'],
                'attention_mask': seq_in['attention_mask'],
            } for seq_in, seq_out in encoded
        ]).with_format('torch')
        loader = DataLoader(
            dataset,
            batch_size=self.predict_batch_size,
            shuffle=False,
            collate_fn=self.collator
        )
        self.model.eval()
        i = 0
        for batch in tqdm.tqdm(loader):
            with pt.no_grad():
                generated = self.model.generate(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    generation_config=self.generation_config
                )
            decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for output_seq in decoded:
                data[i].seq_output = output_seq
                i += 1
        if data.file is not None:
            data.file = pathlib.Path(
                self.directory,self.approach,self.experiment,str(self.iteration),
                'predictions',f"{data.file.path.stem}{data.file.format.extensions[0]}"
            )
        return data

    def logits(self, data:SequencesData):
        encoded = self.preprocess(data)
        dataset = Dataset.from_list(
            [
                {
                    'input_ids': seq_in['input_ids'],
                    'attention_mask': seq_in['attention_mask'],
                    'labels': seq_out['input_ids']
                } for seq_in, seq_out in encoded
            ]
        ).with_format('torch')
        loader = DataLoader(
            dataset,
            batch_size=self.predict_batch_size,
            shuffle=False,
            collate_fn=self.collator
        )
        self.model.eval()
        i = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            with pt.no_grad():
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                ).logits
            for logit_seq, label_seq in zip(logits, labels):
                first_idx = pt.sum(label_seq == -100).item()
                logit_seq = logit_seq[first_idx:]
                label_seq = label_seq[first_idx:]
                logit_seq = pt.log_softmax(logit_seq, dim=1)
                token_logits = torch.gather(logit_seq, dim=1, index=label_seq.unsqueeze(1)).squeeze()
                logits_list = token_logits.tolist()
                data[i].seq_logits = logits_list
                i += 1
        if data.file is not None:
            data.file = pathlib.Path(
                self.directory,self.approach,self.experiment,str(self.iteration),
                'logits',f"{data.file.path.stem}{data.file.format.extensions[0]}"
            )
        return data

    def training(self, data:SequencesData):
        self.record_hyperparameters('training', data=repr(data))
        encoded = self.preprocess(data)
        dataset = Dataset.from_list([
            {
                'input_ids': seq_in['input_ids'],
                'attention_mask': seq_in['attention_mask'],
                'labels': seq_out['input_ids']
            } for seq_in, seq_out in encoded
        ]).with_format('torch')
        del encoded # RAM demanding, free memory
        self.model.train()
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            data_collator=self.collator,
            train_dataset=dataset,
            eval_dataset=dataset,
            callbacks=[StopAfterOneEpoch()],
        )
        for i in range(self.epochs):
            self.iteration += 1
            with ez.shush() if self.silence_training else contextlib.nullcontext():
                self.trainer.train()
            results = SequenceResults(
                epoch=self.iteration,
                loss=self.trainer.state.log_history[-1]['train_loss']
            )
            if data.file is not None:
                results.file = pathlib.Path(
                    self.directory,self.approach,self.experiment,str(self.iteration),
                    'results',f"{data.file.path.stem}{type(results).extensions[0]}"
                )
            yield results

    def train(self, data:SequencesData):
        list(self.training(data))



if __name__ == '__main__':
    t5 = T5(
        experiment='MyExpt',
        epochs=60,

    )