# Frame Examples

A collection of examples using `frame` for language modeling.

### `frame_seq2seq.py`

Seq2Seq summarization model mapping sample sentences from framenet to frame definitions.


This first requires preprocessing the framenet data using `frame`'s cli:

```python
python -m frame.cli preprocess-framenet <path/to/fndata-1.7> <root/path/to/save/preprocessed/data>
```

We can then train the Seq2Seq model with

```python
python frame_seq2seq.py --data <root/path/to/save/preprocessed/data>
```
