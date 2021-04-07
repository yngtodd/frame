from torch.utils.data import Dataset
from nltk.corpus.reader import framenet


class FrameSeq2Seq(Dataset):
    """Map sentences to frame definitions

    For each sentence, return a dictionary that
    contains the sentence text, the associated 
    frame definition, and the frame name.

    Note:
        Framenet's frame definitions can contain 
        a summary definition and examples. A robust
        pipeline should be able to extract these 
        examples, leaving only the necessary definition
        bits.

    Args:
        path: path to the `fndata-1.7` dataset
    """

    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.sentences = self._load_framenet()

    def _load_framenet(self, path):
        """Load framenet sentences generator"""
        fn = framenet.FramenetCorpusReader(self.path, fileids=None)
        return fn.sents()

    def __repr__(self):
        return f"FrameSeq2Seq(path={self.path})"

    def __len__(self):
        return 100_000

    def __getitem__(self, idx):
        return {
            "sentence": self.sentences[idx].text,
            "frame_definition": self.sentences[idx].frame.definition,
            "frame": self.sentences[idx].frame.name
        }
