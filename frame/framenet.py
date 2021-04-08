import csv

from tqdm import tqdm
from pathlib import Path
from nltk.corpus.reader import framenet


def load_sentences(path: str):
    """Load Framenet sentence generator"""
    fn = framenet.FramenetCorpusReader(path, fileids=None)
    return fn.sents()


def save_sentence(sentence, filename):
    """Save sentence and frame information"""
    with open(filename, mode='w') as f:
        fieldnames = ['frame', 'sentence', 'frame_definition']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        writer.writerow({
            "frame": sentence.frame.name, 
            "sentence": sentence.text, 
            "frame_definition": sentence.frame.definition
        })


def save_sentence_data(framenet_path, save_root, num_samples=100_000):
    """Save sentence data as individual files

    Every sample has two documents, a sentence from 
    Framenet as well as the frame definition. This 
    can be used for text summarization Seq2Seq models.
    
    Note:
        As a proof of concept, this only saves the first
        `num_samples`.

    Args:
        framenet_path: path to the `fndata-1.7` dataset
        save_root: root path to save individual sentence csvs
        num_samples: number of samples to save
    """
    sentences = load_sentences(framenet_path)

    root = Path(save_root)
    root.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(num_samples)):
        sent = sentences[idx]
        # Name the file based on frame type
        path = root.joinpath(f"{sent.frame.name}_{idx}.csv")
        save_sentence(sent, path)


def load_sentence(filename):
    """Load a single sentence, frame definition pair"""
    with open(filename, mode='r') as f:
        reader = csv.DictReader(f)

        # Each file we processed is just a single dicionary,
        # we can grab that and run.
        for line in reader:
            sentence = dict(line)

    return sentence


def load_sentence_data(save_root):
    """Load all raww sentence, frame definition pairs

    We will get a list of dictionaries, each
    representing a raw text sentence, frame 
    definition pair.
    """
    root = Path(save_root).glob("**/*")
    files = [x for x in root if x.is_file()]

    sentences = []
    for file in tqdm(files):
        sentences.append(
            load_sentence(file)
        )

    return sentences

