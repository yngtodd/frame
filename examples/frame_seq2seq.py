"""Seq2Seq Summarization - Sentences to Frame Definitions"""
import nltk
import numpy as np

from argparse import ArgumentParser
from frame.framenet import data_paths
from datasets import load_dataset, DatasetDict

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
)


def parse_args():
    parser = ArgumentParser(description="Seq2Seq Example")
    parser.add_argument("--data", type=str, 
                        help="Root path to processed data")
    parser.add_argument("--model", type=str, default="t5-small",
                        help="Root path to processed data")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="max sentence length")
    parser.add_argument("--max_target_length", type=int, default=128,
                        help="max frame definition length")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size for training and eval")
    return parser.parse_args()


def main():
    args = parse_args()

    # this requires having preprocessed framenet
    # using `frame.cli:preprocess-framenet
    paths = data_paths(args.data)
    dataset = load_dataset('json', data_files=paths)

    train_test = dataset["train"].train_test_split(test_size=0.1)
    test_valid = train_test["test"].train_test_split(test_size=0.5)

    datasets = DatasetDict({
        "train": train_test["train"],
        "test": test_valid["test"],
        "valid": test_valid["train"]
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # the family of t5 models expect input sentences to be prefixed with `"summarize: "`
    if args.model in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
        prefix = "summarize: "
    else:
        prefix = ""

    # HuggingFace loves to use closures, I would prefer this 
    # be refactored into the library, but going this route for simplicity
    def preprocess_function(examples):
        """Tokenize the data for Seq2Seq

        Maps over all the examples in the dataset 
        to tokenize both the input framenet sentences
        and the target frame definitions.

        Args:
            examples: samples in the dataset
        """
        inputs = [prefix + sent for sent in examples["sentence"]]

        model_inputs = tokenizer(
            inputs, 
            max_length=args.max_input_length, 
            truncation=True
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["frame_definition"], 
                max_length=args.max_target_length, 
                truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    # again, with the closures - requires instance of the tokenizer
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    tokenized_datasets = datasets.map(preprocess_function, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        "./results/summarization",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()


if __name__=="__main__":
    main()
