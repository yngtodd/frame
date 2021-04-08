"""Seq2Seq Summarization - Sentences to Frame Definitions"""
from datasets import load_dataset
from argparse import ArgumentParser
from frame.framenet import data_paths
from transformers import AutoTokenizer


def parse_args():
    parser = ArgumentParser(description="Seq2Seq Example")
    parser.add_argument("--data", type=str, 
                        help="Root path to processed data")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="max sentence length")
    parser.add_argument("--max_target_length", type=int, default=128,
                        help="max frame definition length")
    return parser.parse_args()


def main():
    args = parse_args()

    # this requires having preprocessed framenet
    # using `frame.cli:preprocess-framenet
    data_paths = data_paths(args.data)
    dataset = load_dataset('json', data_files=data_paths)

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
        inputs = [sent for sent in examples["sentence"]]

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
