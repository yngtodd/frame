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

`[frame_seq2seq](examples/frame_seq2seq.py)`: 

Seq2Seq training mapping framenet sentences to semantic frame definitions using framenet.

#### Notebooks

`[01_base_pipeline](examples/notebooks/01_base_pipeline.ipynb)`: 

Baseline testing summarization of a HuggingFace Seq2Seq model on framenet

`[02_frame_seq2seq_dataset](examples/notebooks/02_frame_seq2seq_dataset.ipynb)`: 

Example usage of a Pytorch Dataset for loading pairs of framenet sentences and frame definitions.

`[03_tokenization](examples/notebooks/03_tokenization.ipynb)`: 

Example showing Seq2Seq tokenization process using HuggingFace's Datasets library. Note: this 
depends on the preprocessed data created by `frame.cli:preprocess-framenet`.

