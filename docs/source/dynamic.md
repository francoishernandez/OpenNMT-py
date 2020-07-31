# Training with dynamic data transform

In contrary to previous method of training models on preprocessed deterministic dataset, we sometimes want multiply variation for the same training example to augment original data. With previous setting, we need to preprocess and dump the dataset as many times as the number of variation we expected. This will consume a lot of disk usage, therefore inefficient.

The needs of data augmentation as a mesure of regularization has been demonstrated in numbers of reaserches. Some can be applied without changes to dataloader, but others requires a larger change to the sequence, makes it impossible to integrete with actual settings, for instance, subword regularization.

We propose therefore a alternative mecanism to current dataloader: dynamic data iterator. This mecanism enables us to train models directely from plain text by applying sequence transforms when loading every lines. This makes it possible to generate different variant for the same sequence by appling stochstic transform.

## Quickstart

To train models with this dynamic feature, a `-data_config` yaml file is required. This config file contains all options related to data transfrom, vocabulary and corpora definition, and need to be used through out the preprocess and trainig phrase. A example setting can be seen at [example](https://github.com/OpenNMT/OpenNMT-py/blob/master/examples/onmt.dynamic.data_config.yaml).

### Step 0: Build vocabulary and data config file

As the data is supposed to be dynamicly processed when iterate, we can not build vocabulary automaticly by walking through the corpora as sequences may not be the same token level before/after transform. Therefore, a vocabulary file is required.

To get the needed vocabulary, if [sentencepiece](https://github.com/google/sentencepiece) is used as subword tokenizer, its learned vocab file can be used and feed to [spm_to_vocab](https://github.com/OpenNMT/OpenNMT-py/blob/master/tools/spm_to_vocab.py) to get the vocab file in OpenNMT-py format. While a vocab file for official bpe by [subword-nmt](https://github.com/rsennrich/subword-nmt) can only be get by walking through the bpe tokenized corpora as there is no vocab file generated when learning.

### Step 1: Preprocess the data

A example cmd for dynamic preprocess:

```bash
python3 OpenNMT-py/onmt/dynamic/preprocess.py -data_config examples/onmt.dynamic.data_config.yaml
```

After runing preprocess, corpora should be linked to `-save_data` together with traditional `*vocab.pt` field file and the new `*transform.pt` containing transforms to be applied to corpora when training.

### Step 2: Training with dynamic transformed data

To lanch the training, another config file for training related options is required.

```bash
python3 OpenNMT-py/onmt/dynamic/train.py -data_config examples/onmt.dynamic.data_config.yaml -config examples/onmt.train.deep8fp16.yaml
```

During training, transform statistics will be reported at the end of each file's iteration. One should keep in mind that, as we process the data online, the more sophisticated transform we use, the more costly for data generation. Therefore, we recommend to only use it in distributed mode and only necessary transforms. However, you can rest assured that using already processed plain text corpora with simple transform like `filteroutlong`, training will not slow down compared to previous binarized mode.

### Step 3: Translating

For translating part, the standard `translate.py` can always be used as before.
