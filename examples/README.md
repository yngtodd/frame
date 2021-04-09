# Frame Examples

A collection of examples using `frame` for language modeling.

### `frame_seq2seq.py`

Seq2Seq summarization model mapping sample sentences from framenet to frame definitions.


This first requires preprocessing the framenet data using `frame`'s cli:

```python
python -m frame.cli preprocess-framenet <path/to/fndata-1.7> <root/path/to/save/preprocessed/data>
```

The evaluation metric for the model also requires a dependency from `nltk`:

```python
import nltk
nltk.download('punkt')
```

We can then train the Seq2Seq model, using the [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric)) 
metric as our litmus test.

```python
python frame_seq2seq.py --data <root/path/to/save/preprocessed/data>
```

### Notes

`t5` seems to be very sensitive to its input data. Evaluating the model after fine tuning for only 
a handful of epochs, it seems that the model's ability to summarize text changes quickly. 

Things to try:

- [ ] longer trainer runs (currently training 100 epochs on a single Quadro RTX 6000.
- [ ] preprocessing the framenet frame definitions. Are these really the best summaries for the sample sentences? 
