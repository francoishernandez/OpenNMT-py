
# Library

The example notebook (available [here](https://github.com/OpenNMT/OpenNMT-py/blob/master/docs/source/Library.ipynb)) should be able to run as a standalone execution, provided `onmt` is in the path (installed via `pip` for instance).

Some parts may not be 100% 'library-friendly' but it's mostly workable.

### Import a few modules and functions that will be necessary


```python
import yaml
import torch
import torch.nn as nn
from argparse import Namespace
from collections import defaultdict, Counter
```


```python
import onmt
from onmt.inputters.inputter import _load_vocab, _build_fields_vocab, get_fields, IterOnDevice
from onmt.dynamic.corpus import ParallelCorpus
from onmt.dynamic.iterator import DynamicDatasetIter
from onmt.translate import GNMTGlobalScorer, Translator, TranslationBuilder
```

### Enable logging


```python
# enable logging
from onmt.utils.logging import init_logger, logger
init_logger()
```




    <RootLogger root (INFO)>



### Retrieve data

To make a proper example, we will need some data, as well as some vocabulary(ies).

Let's take the same data as in the [quickstart](https://opennmt.net/OpenNMT-py/quickstart.html):


```python
!wget https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz
```

    --2020-09-22 16:31:08--  https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz
    Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.216.97.181
    Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.216.97.181|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1662081 (1,6M) [application/x-gzip]
    Saving to: ‘toy-ende.tar.gz.15’
    
    toy-ende.tar.gz.15  100%[===================>]   1,58M  2,17MB/s    in 0,7s    
    
    2020-09-22 16:31:09 (2,17 MB/s) - ‘toy-ende.tar.gz.15’ saved [1662081/1662081]
    



```python
!tar xf toy-ende.tar.gz
```


```python
ls toy-ende
```

    config.yaml  src-test.txt   src-val.txt   tgt-train.txt
    run          src-train.txt  tgt-test.txt  tgt-val.txt


### Prepare data and vocab

As for any use case of OpenNMT-py 2.0, we can start by creating a simple YAML configuration with our datasets. This is the easiest way to build the proper `opts` `Namespace` that will be used to create the vocabulary(ies).


```python
yaml_config = """
## Where the vocab(s) will be written
save_data: toy-ende/run/example
# Corpus opts:
data:
    corpus:
        path_src: toy-ende/src-train.txt
        path_tgt: toy-ende/tgt-train.txt
        transforms: []
        weight: 1
    valid:
        path_src: toy-ende/src-val.txt
        path_tgt: toy-ende/tgt-val.txt
        transforms: []
"""
config = yaml.safe_load(yaml_config)
with open("toy-ende/config.yaml", "w") as f:
    f.write(yaml_config)
```


```python
from onmt.dynamic.parse import DynamicArgumentParser
parser = DynamicArgumentParser(description='build_vocab.py')
```


```python
from onmt.dynamic.opts import dynamic_prepare_opts
dynamic_prepare_opts(parser, build_vocab_only=True)
```


```python
base_args = (["-config", "toy-ende/config.yaml", "-n_sample", "10000"])
opts, unknown = parser.parse_known_args(base_args)
```


```python
opts
```




    Namespace(config='toy-ende/config.yaml', data="{'corpus': {'path_src': 'toy-ende/src-train.txt', 'path_tgt': 'toy-ende/tgt-train.txt', 'transforms': [], 'weight': 1}, 'valid': {'path_src': 'toy-ende/src-val.txt', 'path_tgt': 'toy-ende/tgt-val.txt', 'transforms': []}}", insert_ratio=0.0, mask_length='subword', mask_ratio=0.0, n_sample=10000, onmttok_kwargs="{'mode': 'none'}", overwrite=False, permute_sent_ratio=0.0, poisson_lambda=0.0, random_ratio=0.0, replace_length=-1, rotate_ratio=0.5, save_config=None, save_data='toy-ende/run/example', seed=-1, share_vocab=False, src_seq_length=200, src_subword_model=None, src_subword_type='none', src_vocab=None, subword_alpha=0, subword_nbest=1, switchout_temperature=1.0, tgt_seq_length=200, tgt_subword_model=None, tgt_subword_type='none', tgt_vocab=None, tokendrop_temperature=1.0, tokenmask_temperature=1.0, transforms=[])




```python
from onmt.bin.build_vocab import build_vocab_main
build_vocab_main(opts)
```

    [2020-09-22 16:31:09,862 WARNING] Corpus valid's weight should be given. We default it to 1 for you.
    [2020-09-22 16:31:09,866 INFO] Parsed 2 corpora from -data.
    [2020-09-22 16:31:09,868 INFO] Counter vocab from 10000 samples.
    [2020-09-22 16:31:09,871 INFO] corpus's transforms: TransformPipe()
    [2020-09-22 16:31:09,901 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-22 16:31:10,084 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-22 16:31:10,115 INFO] Counters src:24995
    [2020-09-22 16:31:10,116 INFO] Counters tgt:35816



```python
ls toy-ende/run
```

    example.vocab.src  example.vocab.tgt  sample


We just created our source and target vocabularies, respectively `toy-ende/run/example.vocab.src` and `toy-ende/run/example.vocab.tgt`.

### Build fields

We can build the fields from the text files that were just created.


```python
src_vocab_path = "toy-ende/run/example.vocab.src"
tgt_vocab_path = "toy-ende/run/example.vocab.tgt"
```


```python
# initialize the frequency counter
counters = defaultdict(Counter)
# load source vocab
_src_vocab, _src_vocab_size = _load_vocab(
    src_vocab_path,
    'src',
    counters)
# load target vocab
_tgt_vocab, _tgt_vocab_size = _load_vocab(
    tgt_vocab_path,
    'tgt',
    counters)
```

    [2020-09-22 16:31:11,523 INFO] Loading src vocabulary from toy-ende/run/example.vocab.src
    [2020-09-22 16:31:11,584 INFO] Loaded src vocab has 24995 tokens.
    [2020-09-22 16:31:11,593 INFO] Loading tgt vocabulary from toy-ende/run/example.vocab.tgt
    [2020-09-22 16:31:11,647 INFO] Loaded tgt vocab has 35816 tokens.



```python
# initialize fields
src_nfeats, tgt_nfeats = 0, 0 # do not support word features for now
fields = get_fields(
    'text', src_nfeats, tgt_nfeats)
```


```python
fields
```




    {'src': <onmt.inputters.text_dataset.TextMultiField at 0x7fb613e8b748>,
     'tgt': <onmt.inputters.text_dataset.TextMultiField at 0x7fb613e8b828>,
     'indices': <torchtext.data.field.Field at 0x7fb613e8bf60>}




```python
# build fields vocab
share_vocab = False
vocab_size_multiple = 1
src_vocab_size = 30000
tgt_vocab_size = 30000
src_words_min_frequency = 1
tgt_words_min_frequency = 1
vocab_fields = _build_fields_vocab(
    fields, counters, 'text', share_vocab,
    vocab_size_multiple,
    src_vocab_size, src_words_min_frequency,
    tgt_vocab_size, tgt_words_min_frequency)
```

    [2020-09-22 16:31:12,362 INFO]  * tgt vocab size: 30004.
    [2020-09-22 16:31:12,412 INFO]  * src vocab size: 24997.


An alternative way of creating these fields is to run `onmt_train` without actually training, to just output the necessary files.

### Prepare for training: model and optimizer creation

Let's get a few fields/vocab related variables to simplify the model creation a bit:


```python
src_text_field = vocab_fields["src"].base_field
src_vocab = src_text_field.vocab
src_padding = src_vocab.stoi[src_text_field.pad_token]

tgt_text_field = vocab_fields['tgt'].base_field
tgt_vocab = tgt_text_field.vocab
tgt_padding = tgt_vocab.stoi[tgt_text_field.pad_token]
```

Next we specify the core model itself. Here we will build a small model with an encoder and an attention based input feeding decoder. Both models will be RNNs and the encoder will be bidirectional


```python
emb_size = 100
rnn_size = 500
# Specify the core model.

encoder_embeddings = onmt.modules.Embeddings(emb_size, len(src_vocab),
                                             word_padding_idx=src_padding)

encoder = onmt.encoders.RNNEncoder(hidden_size=rnn_size, num_layers=1,
                                   rnn_type="LSTM", bidirectional=True,
                                   embeddings=encoder_embeddings)

decoder_embeddings = onmt.modules.Embeddings(emb_size, len(tgt_vocab),
                                             word_padding_idx=tgt_padding)
decoder = onmt.decoders.decoder.InputFeedRNNDecoder(
    hidden_size=rnn_size, num_layers=1, bidirectional_encoder=True, 
    rnn_type="LSTM", embeddings=decoder_embeddings)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = onmt.models.model.NMTModel(encoder, decoder)
model.to(device)

# Specify the tgt word generator and loss computation module
model.generator = nn.Sequential(
    nn.Linear(rnn_size, len(tgt_vocab)),
    nn.LogSoftmax(dim=-1)).to(device)

loss = onmt.utils.loss.NMTLossCompute(
    criterion=nn.NLLLoss(ignore_index=tgt_padding, reduction="sum"),
    generator=model.generator)
```

Now we set up the optimizer. This could be a core torch optim class, or our wrapper which handles learning rate updates and gradient normalization automatically.


```python
lr = 1
torch_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optim = onmt.utils.optimizers.Optimizer(
    torch_optimizer, learning_rate=lr, max_grad_norm=2)
```

### Create the training and validation data iterators

Now we need to create the dynamic dataset iterator.

This is not very 'library-friendly' for now because of the way the `DynamicDatasetIter` constructor is defined. It may evolve in the future.


```python
src_train = "toy-ende/src-train.txt"
tgt_train = "toy-ende/tgt-train.txt"
src_val = "toy-ende/src-val.txt"
tgt_val = "toy-ende/tgt-val.txt"

# build the ParallelCorpus
corpus = ParallelCorpus(src_train, tgt_train)
valid = ParallelCorpus(src_val, tgt_val)
```


```python
corpora = {"corpus": corpus}
transforms = {}
opts = Namespace()
opts.batch_size = 4096
opts.batch_type = "tokens"
opts.valid_batch_size = 8
opts.batch_size_multiple = 1
opts.data_type = "text"
opts.bucket_size = 4096
opts.pool_factor = 100
opts.data = {"corpus": {"weight": 1}}
```


```python
# build the training iterator
is_train = True
train_iter = DynamicDatasetIter(
    corpora, transforms, vocab_fields, opts, is_train,
    stride=1, offset=0)
```


```python
# make sure the iteration happens on GPU 0 (-1 for CPU, N for GPU N)
train_iter = iter(IterOnDevice(train_iter, 0))
```


```python
corpora = {"valid": valid}
transforms = {}
opts = Namespace()
opts.batch_size = 4096
opts.batch_type = "tokens"
opts.valid_batch_size = 8
opts.batch_size_multiple = 1
opts.data_type = "text"
opts.bucket_size = 4096
opts.pool_factor = 100
opts.data = {"valid": {"weight": 1}}
```


```python
# build the validation iterator
is_train = False
valid_iter = DynamicDatasetIter(
    corpora, transforms, vocab_fields, opts, is_train,
    stride=1, offset=0)
```


```python
valid_iter = IterOnDevice(valid_iter, 0)
```

### Training

Finally we train.


```python
report_manager = onmt.utils.ReportMgr(
    report_every=50, start_time=None, tensorboard_writer=None)

trainer = onmt.Trainer(model=model,
                       train_loss=loss,
                       valid_loss=loss,
                       optim=optim,
                       report_manager=report_manager,
                       dropout=[0.1])

trainer.train(train_iter=train_iter,
              train_steps=1000,
              valid_iter=valid_iter,
              valid_steps=500)
```

    [2020-09-22 16:31:18,569 INFO] Start training loop and validate every 500 steps...
    [2020-09-22 16:31:18,572 INFO] corpus's transforms: TransformPipe()
    [2020-09-22 16:31:18,574 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-22 16:31:24,717 INFO] Step 50/ 1000; acc:   7.21; ppl: 6659.24; xent: 8.80; lr: 1.00000; 19307/19183 tok/s;      6 sec
    [2020-09-22 16:31:28,225 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-22 16:31:30,779 INFO] Step 100/ 1000; acc:   9.26; ppl: 1989.38; xent: 7.60; lr: 1.00000; 19281/19312 tok/s;     12 sec
    [2020-09-22 16:31:36,770 INFO] Step 150/ 1000; acc:  10.16; ppl: 1374.93; xent: 7.23; lr: 1.00000; 18835/18653 tok/s;     18 sec
    [2020-09-22 16:31:37,902 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-22 16:31:42,944 INFO] Step 200/ 1000; acc:  11.11; ppl: 1114.67; xent: 7.02; lr: 1.00000; 19001/18907 tok/s;     24 sec
    [2020-09-22 16:31:49,075 INFO] Step 250/ 1000; acc:  12.05; ppl: 940.74; xent: 6.85; lr: 1.00000; 19266/19120 tok/s;     31 sec
    [2020-09-22 16:31:52,315 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-22 16:31:55,233 INFO] Step 300/ 1000; acc:  13.13; ppl: 756.69; xent: 6.63; lr: 1.00000; 18918/18918 tok/s;     37 sec
    [2020-09-22 16:32:01,301 INFO] Step 350/ 1000; acc:  13.84; ppl: 673.48; xent: 6.51; lr: 1.00000; 18444/18335 tok/s;     43 sec
    [2020-09-22 16:32:02,179 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-22 16:32:07,458 INFO] Step 400/ 1000; acc:  14.74; ppl: 579.51; xent: 6.36; lr: 1.00000; 19241/19129 tok/s;     49 sec
    [2020-09-22 16:32:13,644 INFO] Step 450/ 1000; acc:  16.07; ppl: 507.73; xent: 6.23; lr: 1.00000; 18889/18905 tok/s;     55 sec
    [2020-09-22 16:32:16,765 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-22 16:32:19,841 INFO] Step 500/ 1000; acc:  16.48; ppl: 456.66; xent: 6.12; lr: 1.00000; 19076/18925 tok/s;     61 sec
    [2020-09-22 16:32:19,842 INFO] valid's transforms: TransformPipe()
    [2020-09-22 16:32:19,844 INFO] Loading ParallelCorpus(toy-ende/src-val.txt, toy-ende/tgt-val.txt, align=None)...
    [2020-09-22 16:32:28,779 INFO] Validation perplexity: 276.766
    [2020-09-22 16:32:28,780 INFO] Validation accuracy: 20.0439
    [2020-09-22 16:32:34,824 INFO] Step 550/ 1000; acc:  17.48; ppl: 404.68; xent: 6.00; lr: 1.00000; 7507/7418 tok/s;     76 sec
    [2020-09-22 16:32:35,565 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-22 16:32:41,112 INFO] Step 600/ 1000; acc:  18.90; ppl: 348.96; xent: 5.85; lr: 1.00000; 18067/18046 tok/s;     83 sec
    [2020-09-22 16:32:47,363 INFO] Step 650/ 1000; acc:  19.69; ppl: 311.28; xent: 5.74; lr: 1.00000; 18561/18595 tok/s;     89 sec
    [2020-09-22 16:32:50,417 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-22 16:32:53,650 INFO] Step 700/ 1000; acc:  20.55; ppl: 279.91; xent: 5.63; lr: 1.00000; 18735/18615 tok/s;     95 sec
    [2020-09-22 16:32:59,905 INFO] Step 750/ 1000; acc:  21.90; ppl: 243.30; xent: 5.49; lr: 1.00000; 18580/18489 tok/s;    101 sec
    [2020-09-22 16:33:00,493 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-22 16:33:06,243 INFO] Step 800/ 1000; acc:  23.01; ppl: 215.40; xent: 5.37; lr: 1.00000; 17972/17859 tok/s;    108 sec
    [2020-09-22 16:33:10,432 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-22 16:33:12,454 INFO] Step 850/ 1000; acc:  24.39; ppl: 188.56; xent: 5.24; lr: 1.00000; 18979/18915 tok/s;    114 sec
    [2020-09-22 16:33:18,704 INFO] Step 900/ 1000; acc:  25.39; ppl: 169.35; xent: 5.13; lr: 1.00000; 18709/18341 tok/s;    120 sec
    [2020-09-22 16:33:24,976 INFO] Step 950/ 1000; acc:  26.46; ppl: 150.88; xent: 5.02; lr: 1.00000; 18393/18571 tok/s;    126 sec
    [2020-09-22 16:33:25,388 INFO] Loading ParallelCorpus(toy-ende/src-train.txt, toy-ende/tgt-train.txt, align=None)...
    [2020-09-22 16:33:31,180 INFO] Step 1000/ 1000; acc:  27.87; ppl: 132.36; xent: 4.89; lr: 1.00000; 18655/18517 tok/s;    133 sec
    [2020-09-22 16:33:31,182 INFO] Loading ParallelCorpus(toy-ende/src-val.txt, toy-ende/tgt-val.txt, align=None)...
    [2020-09-22 16:33:40,079 INFO] Validation perplexity: 218.311
    [2020-09-22 16:33:40,080 INFO] Validation accuracy: 21.7181





    <onmt.utils.statistics.Statistics at 0x7fb613a1ba90>



### Translate

For translation, we can build a "traditional" (as opposed to dynamic) dataset for now.


```python
src_data = {"reader": onmt.inputters.str2reader["text"](), "data": src_val}
tgt_data = {"reader": onmt.inputters.str2reader["text"](), "data": tgt_val}
_readers, _data = onmt.inputters.Dataset.config(
    [('src', src_data), ('tgt', tgt_data)])
```


```python
dataset = onmt.inputters.Dataset(
    vocab_fields, readers=_readers, data=_data,
    sort_key=onmt.inputters.str2sortkey["text"])
```


```python
data_iter = onmt.inputters.OrderedIterator(
            dataset=dataset,
            device="cuda",
            batch_size=10,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )
```


```python
src_reader = onmt.inputters.str2reader["text"]
tgt_reader = onmt.inputters.str2reader["text"]
scorer = GNMTGlobalScorer(alpha=0.7, 
                          beta=0., 
                          length_penalty="avg", 
                          coverage_penalty="none")
gpu = 0 if torch.cuda.is_available() else -1
translator = Translator(model=model, 
                        fields=vocab_fields, 
                        src_reader=src_reader, 
                        tgt_reader=tgt_reader, 
                        global_scorer=scorer,
                        gpu=gpu)
builder = onmt.translate.TranslationBuilder(data=dataset, 
                                            fields=vocab_fields)
```

**Note**: translations will be very poor, because of the very low quantity of data, the absence of proper tokenization, and the brevity of the training.


```python
for batch in data_iter:
    trans_batch = translator.translate_batch(
        batch=batch, src_vocabs=[src_vocab],
        attn_debug=False)
    translations = builder.from_batch(trans_batch)
    for trans in translations:
        print(trans.log(0))
    break
```

    
    SENT 0: ['Parliament', 'Does', 'Not', 'Support', 'Amendment', 'Freeing', 'Tymoshenko']
    PRED 0: Das Parlament , die Europäische Parlament , die dem Parlament , dem Parlament , dem Parlament , dem Parlament , dem Parlament zu <unk> .
    PRED SCORE: -1.6008
    
    
    SENT 0: ['Today', ',', 'the', 'Ukraine', 'parliament', 'dismissed', ',', 'within', 'the', 'Code', 'of', 'Criminal', 'Procedure', 'amendment', ',', 'the', 'motion', 'to', 'revoke', 'an', 'article', 'based', 'on', 'which', 'the', 'opposition', 'leader', ',', 'Yulia', 'Tymoshenko', ',', 'was', 'sentenced', '.']
    PRED 0: Die Aussprache , die sich in den letzten Jahren , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage ,
    PRED SCORE: -1.7200
    
    
    SENT 0: ['The', 'amendment', 'that', 'would', 'lead', 'to', 'freeing', 'the', 'imprisoned', 'former', 'Prime', 'Minister', 'was', 'revoked', 'during', 'second', 'reading', 'of', 'the', 'proposal', 'for', 'mitigation', 'of', 'sentences', 'for', 'economic', 'offences', '.']
    PRED 0: Die Tatsache , die im Rahmen war , war es , die in der Lage , die im Vorschlag , war .
    PRED SCORE: -1.6019
    
    
    SENT 0: ['In', 'October', ',', 'Tymoshenko', 'was', 'sentenced', 'to', 'seven', 'years', 'in', 'prison', 'for', 'entering', 'into', 'what', 'was', 'reported', 'to', 'be', 'a', 'disadvantageous', 'gas', 'deal', 'with', 'Russia', '.']
    PRED 0: In einigen Jahren war es , um zu einem Jahren zu <unk> , was die Zusammenarbeit mit Russland .
    PRED SCORE: -1.5860
    
    
    SENT 0: ['The', 'verdict', 'is', 'not', 'yet', 'final;', 'the', 'court', 'will', 'hear', 'Tymoshenko', '&apos;s', 'appeal', 'in', 'December', '.']
    PRED 0: Die Aussprache ist nicht <unk> , die <unk> <unk> und <unk> <unk> .
    PRED SCORE: -1.4980
    
    
    SENT 0: ['Tymoshenko', 'claims', 'the', 'verdict', 'is', 'a', 'political', 'revenge', 'of', 'the', 'regime;', 'in', 'the', 'West', ',', 'the', 'trial', 'has', 'also', 'evoked', 'suspicion', 'of', 'being', 'biased', '.']
    PRED 0: Es ist eine Frage der <unk> , in der Nähe , die in der Lage , die sich in der Lage , die in der Lage , in denen auch <unk> .
    PRED SCORE: -1.8194
    
    
    SENT 0: ['The', 'proposal', 'to', 'remove', 'Article', '365', 'from', 'the', 'Code', 'of', 'Criminal', 'Procedure', ',', 'upon', 'which', 'the', 'former', 'Prime', 'Minister', 'was', 'sentenced', ',', 'was', 'supported', 'by', '147', 'members', 'of', 'parliament', '.']
    PRED 0: Der Vorschlag , die sich von den Menschen , die in der Türkei , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die in der Lage , die
    PRED SCORE: -1.6393
    
    
    SENT 0: ['Its', 'ratification', 'would', 'require', '226', 'votes', '.']
    PRED 0: Bitte beachten Sie , dass sie <unk> werden .
    PRED SCORE: -1.4621
    
    
    SENT 0: ['Libya', '&apos;s', 'Victory']
    PRED 0: In der Nähe ist es , wenn man eine <unk> , <unk> , <unk> .
    PRED SCORE: -1.9260
    
    
    SENT 0: ['The', 'story', 'of', 'Libya', '&apos;s', 'liberation', ',', 'or', 'rebellion', ',', 'already', 'has', 'its', 'defeated', '.']
    PRED 0: Die Firma , <unk> oder <unk> , <unk> , <unk> , <unk> , <unk> .
    PRED SCORE: -1.6742
