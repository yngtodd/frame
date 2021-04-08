<h1>Frame<img src='https://github.com/yngtodd/frame/blob/main/img/snek.png' align='right' width='180' height='104'></h1>

Seq2Seq modeling for frame semantics.

#### Setup

```
python setup.py install
```

#### Features

Frame comes with a built in command line interface:

`preprocess-framenet`: save sentences and corresponding frame definitions as csv files. This 
can then be used by HuggingFace's Datasets library to prepare the data for Seq2Seq models.

```python
python -m frame.cli preprocess-framenet <path/to/fndata-1.7> <root/path/to/save/preprocessed/data>
```

#### Examples

[frame_seq2seq](examples/frame_seq2seq.py): Seq2Seq training mapping framenet sentences to 
semantic frame definitions using framenet.


#### Notebooks


