"""Module that contain iterator used for dynamic data."""
from itertools import cycle

from torchtext.data import Dataset as TorchtextDataset, \
    Example as TorchtextExample, batch as torchtext_batch
from onmt.inputters import str2sortkey
from onmt.inputters.inputter import max_tok_len, OrderedIterator
from onmt.utils.logging import logger

from onmt.dynamic.shard import get_corpora_shards, build_sharded_corpora_iters
from onmt.dynamic.transform import load_transforms


class DatasetAdapter(object):
    """Adapte a buckets of tuples into examples of a torchtext Dataset."""

    valid_field_name = ('src', 'tgt', 'indices')

    def __init__(self, fields):
        self._valid_fields(fields)

    def _valid_fields(self, fields):
        valid_fields = []
        for f_name in self.valid_field_name:
            valid_fields.append((f_name, fields[f_name]))
        self.fields_list = valid_fields

    def _to_examples(self, bucket):
        examples = []
        for (src, tgt, index) in bucket:
            ex = TorchtextExample.fromlist((src, tgt, index), self.fields_list)
            examples.append(ex)
        return examples

    def __call__(self, bucket):
        examples = self._to_examples(bucket)
        dataset = TorchtextDataset(examples, self.fields_list)
        return dataset


class DynamicDatasetIter(object):
    """Yield data from multiply sharded plain text files."""

    def __init__(self, corpora_shards, transforms, fields, opts, is_train):
        self.corpora_shards = corpora_shards
        self.transforms = transforms
        self.fields = fields
        self.corpora_info = opts.data
        self.is_train = is_train
        self._init_datasets()

        self.batch_size = opts.batch_size if is_train \
            else opts.valid_batch_size
        self.batch_size_fn = max_tok_len \
            if opts.batch_type == "tokens" else None
        if opts.batch_size_multiple is not None:
            self.batch_size_multiple = opts.batch_size_multiple
        else:
            self.batch_size_multiple = 8 if opts.model_dtype == "fp16" else 1
        self.device = 'cpu'
        self.sort_key = str2sortkey[opts.data_type]
        self.bucket_size = opts.bucket_size
        self.pool_factor = opts.pool_factor

    def _init_datasets(self):
        self.datasets_iterables = build_sharded_corpora_iters(
            self.corpora_shards, self.transforms,
            self.corpora_info, self.is_train)
        self.dataset_adapter = DatasetAdapter(self.fields)
        self.datasets_weights = {
            ds_name: int(self.corpora_info[ds_name]['weight'])
            for ds_name in self.datasets_iterables.keys()
        }
        self.datasets_iters = {
            ds_name: iter(generator)
            for ds_name, generator in self.datasets_iterables.items()
        }

    def _iter_datasets(self):
        """Yield a bucket of examples from weighted datasets."""
        # import pdb; pdb.set_trace()
        for ds_name, ds_weight in self.datasets_weights.items():
            for _ in range(ds_weight):
                yield self.datasets_iters[ds_name]

    def _iter_examples(self):
        for iterator in cycle(self._iter_datasets()):
            yield next(iterator)

    def _bucketing(self):
        buckets = torchtext_batch(
            self._iter_examples(),
            batch_size=self.bucket_size,
            batch_size_fn=None)
        yield from buckets

    def __iter__(self):
        while True:
            for bucket in self._bucketing():
                dataset = self.dataset_adapter(bucket)
                train_iter = OrderedIterator(
                    dataset,
                    self.batch_size,
                    pool_factor=self.pool_factor,
                    batch_size_fn=self.batch_size_fn,
                    batch_size_multiple=self.batch_size_multiple,
                    device=self.device,
                    train=self.is_train,
                    sort=False,
                    sort_within_batch=True,
                    sort_key=self.sort_key,
                    repeat=False,
                )
                for batch in train_iter:
                    # if mb_callback is not None:
                    #     mb_callback(i)
                    yield batch
                    # i += 1


def build_dynamic_dataset_iter(fields, opts, is_train=True):
    """Build `DynamicDatasetIter` from fields & opts."""
    transforms = load_transforms(opts)
    logger.info("Transforms loaded.")
    corpora_shards = get_corpora_shards(opts, is_train)
    if corpora_shards is None:
        assert not is_train, "only valid corpus is ignorable."
        return None
    return DynamicDatasetIter(
        corpora_shards, transforms, fields, opts, is_train)
