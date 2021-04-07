from torch.utils.data import Dataset
from nltk.corpus.reader import framenet


class FrameSeq2Seq(Dataset):
    """Map sentences to frame definitions"""

    def __init__(path):
        super().__init__()
        sentences = self._load_framenet(path)

    def _load_framenet(self, path):
        fn = framenet.FramenetCorpusReader(path, fileids=None)
        return fn.sents()

    def __getitem__(self, idx):
        return {
            "sentence": self.sentences[i].text,
            "frame_definition": self.sentences[i].definition,
            "frame": self.sentences[i].name
        }
