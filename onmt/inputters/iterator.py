"""Contains all methods relate to iteration."""
import glob
from itertools import cycle

import torch
import torchtext.data
from torchtext.data.utils import RandomShuffler

from onmt.utils.logging import logger


def batch_iter(data, batch_size, batch_size_fn=None, batch_size_multiple=1):
    """Yield elements from data in chunks of batch_size, where each chunk size
    is a multiple of batch_size_multiple.

    This is an extended version of torchtext.data.batch.
    """
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far >= batch_size:
            overflowed = 0
            if size_so_far > batch_size:
                overflowed += 1
            if batch_size_multiple > 1:
                overflowed += (
                    (len(minibatch) - overflowed) % batch_size_multiple)
            if overflowed == 0:
                yield minibatch
                minibatch, size_so_far = [], 0
            else:
                if overflowed == len(minibatch):
                    logger.warning(
                        "The batch will be filled until we reach %d,"
                        "its size may exceed %d tokens"
                        % (batch_size_multiple, batch_size)
                        )
                else:
                    yield minibatch[:-overflowed]
                    minibatch = minibatch[-overflowed:]
                    size_so_far = 0
                    for i, ex in enumerate(minibatch):
                        size_so_far = batch_size_fn(ex, i + 1, size_so_far)
    if minibatch:
        yield minibatch


def _pool(data, batch_size, batch_size_fn, batch_size_multiple,
          sort_key, random_shuffler, pool_factor):
    for p in torchtext.data.batch(
            data, batch_size * pool_factor,
            batch_size_fn=batch_size_fn):
        p_batch = list(batch_iter(
            sorted(p, key=sort_key),
            batch_size,
            batch_size_fn=batch_size_fn,
            batch_size_multiple=batch_size_multiple))
        for b in random_shuffler(p_batch):
            yield b


class OrderedIterator(torchtext.data.Iterator):

    def __init__(self,
                 dataset,
                 batch_size,
                 pool_factor=1,
                 batch_size_multiple=1,
                 yield_raw_example=False,
                 **kwargs):
        super(OrderedIterator, self).__init__(dataset, batch_size, **kwargs)
        self.batch_size_multiple = batch_size_multiple
        self.yield_raw_example = yield_raw_example
        self.dataset = dataset
        self.pool_factor = pool_factor

    def create_batches(self):
        if self.train:
            if self.yield_raw_example:
                self.batches = batch_iter(
                    self.data(),
                    1,
                    batch_size_fn=None,
                    batch_size_multiple=1)
            else:
                self.batches = _pool(
                    self.data(),
                    self.batch_size,
                    self.batch_size_fn,
                    self.batch_size_multiple,
                    self.sort_key,
                    self.random_shuffler,
                    self.pool_factor)
        else:
            self.batches = []
            for b in batch_iter(
                    self.data(),
                    self.batch_size,
                    batch_size_fn=self.batch_size_fn,
                    batch_size_multiple=self.batch_size_multiple):
                self.batches.append(sorted(b, key=self.sort_key))

    def __iter__(self):
        """
        Extended version of the definition in torchtext.data.Iterator.
        Added yield_raw_example behaviour to yield a torchtext.data.Example
        instead of a torchtext.data.Batch object.
        """
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a
                    # minibatch be sorted by decreasing order, which
                    #  requires reversing relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                if self.yield_raw_example:
                    yield minibatch[0]
                else:
                    yield torchtext.data.Batch(
                        minibatch,
                        self.dataset,
                        self.device)
            if not self.repeat:
                return


def max_tok_len(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/tgt tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src and tgt length in the current batch
    global max_src_in_batch, max_tgt_in_batch  # this is a hack
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    # Src: [<bos> w1 ... wN <eos>]
    max_src_in_batch = max(max_src_in_batch, len(new.src[0]) + 2)
    # Tgt: [w1 ... wM <eos>]
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt[0]) + 1)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class DatasetLazyIter(object):
    """Yield data from sharded dataset files.

    Args:
        dataset_paths: a list containing the locations of dataset files.
        fields (dict[str, Field]): fields dict for the
            datasets.
        batch_size (int): batch size.
        batch_size_fn: custom batch process function.
        device: See :class:`OrderedIterator` ``device``.
        is_train (bool): train or valid?
    """

    def __init__(self, dataset_paths, fields, batch_size, batch_size_fn,
                 batch_size_multiple, device, is_train, pool_factor,
                 repeat=True, num_batches_multiple=1, yield_raw_example=False):
        self._paths = dataset_paths
        self.fields = fields
        self.batch_size = batch_size
        self.batch_size_fn = batch_size_fn
        self.batch_size_multiple = batch_size_multiple
        self.device = device
        self.is_train = is_train
        self.repeat = repeat
        self.num_batches_multiple = num_batches_multiple
        self.yield_raw_example = yield_raw_example
        self.pool_factor = pool_factor

    def _iter_dataset(self, path):
        logger.info('Loading dataset from %s' % path)
        cur_dataset = torch.load(path)
        logger.info('number of examples: %d' % len(cur_dataset))
        cur_dataset.fields = self.fields
        cur_iter = OrderedIterator(
            dataset=cur_dataset,
            batch_size=self.batch_size,
            pool_factor=self.pool_factor,
            batch_size_multiple=self.batch_size_multiple,
            batch_size_fn=self.batch_size_fn,
            device=self.device,
            train=self.is_train,
            sort=False,
            sort_within_batch=True,
            repeat=False,
            yield_raw_example=self.yield_raw_example
        )
        for batch in cur_iter:
            self.dataset = cur_iter.dataset
            yield batch

        # NOTE: This is causing some issues for consumer/producer,
        # as we may still have some of those examples in some queue
        # cur_dataset.examples = None
        # gc.collect()
        # del cur_dataset
        # gc.collect()

    def __iter__(self):
        num_batches = 0
        paths = self._paths
        if self.is_train and self.repeat:
            # Cycle through the shards indefinitely.
            paths = cycle(paths)
        for path in paths:
            for batch in self._iter_dataset(path):
                yield batch
                num_batches += 1
        if self.is_train and not self.repeat and \
           num_batches % self.num_batches_multiple != 0:
            # When the dataset is not repeated, we might need to ensure that
            # the number of returned batches is the multiple of a given value.
            # This is important for multi GPU training to ensure that all
            # workers have the same number of batches to process.
            for path in paths:
                for batch in self._iter_dataset(path):
                    yield batch
                    num_batches += 1
                    if num_batches % self.num_batches_multiple == 0:
                        return


def build_dataset_iter(corpus_type, fields, opt, is_train=True, multi=False):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over. We implement simple ordered iterator strategy here,
    but more sophisticated strategy like curriculum learning is ok too.
    """
    dataset_glob = opt.data + '.' + corpus_type + '.[0-9]*.pt'
    dataset_paths = list(sorted(
        glob.glob(dataset_glob),
        key=lambda p: int(p.split(".")[-2])))

    if not dataset_paths:
        if is_train:
            raise ValueError('Training data %s not found' % dataset_glob)
        else:
            return None
    if multi:
        batch_size = 1
        batch_fn = None
        batch_size_multiple = 1
    else:
        batch_size = opt.batch_size if is_train else opt.valid_batch_size
        batch_fn = max_tok_len \
            if is_train and opt.batch_type == "tokens" else None
        batch_size_multiple = 8 if opt.model_dtype == "fp16" else 1

    device = "cpu"

    return DatasetLazyIter(
        dataset_paths,
        fields,
        batch_size,
        batch_fn,
        batch_size_multiple,
        device,
        is_train,
        opt.pool_factor,
        repeat=not opt.single_pass,
        num_batches_multiple=max(opt.accum_count) * opt.world_size,
        yield_raw_example=multi)


class MultipleDatasetIterator(object):
    """
    This takes a list of iterable objects (DatasetLazyIter) and their
    respective weights, and yields a batch in the wanted proportions.
    """
    def __init__(self,
                 train_shards,
                 fields,
                 device,
                 opt):
        self.index = -1
        self.iterables = []
        self.weights = []
        for shard, weight in zip(train_shards, opt.data_weights):
            if weight > 0:
                self.iterables.append(
                    build_dataset_iter(shard, fields, opt, multi=True))
                self.weights.append(weight)
        self.init_iterators = True
        # self.weights = opt.data_weights
        self.batch_size = opt.batch_size
        self.batch_size_fn = max_tok_len \
            if opt.batch_type == "tokens" else None
        if opt.batch_size_multiple is not None:
            self.batch_size_multiple = opt.batch_size_multiple
        else:
            self.batch_size_multiple = 8 if opt.model_dtype == "fp16" else 1
        self.device = device
        # Temporarily load one shard to retrieve sort_key for data_type
        temp_dataset = torch.load(self.iterables[0]._paths[0])
        self.sort_key = temp_dataset.sort_key
        self.random_shuffler = RandomShuffler()
        self.pool_factor = opt.pool_factor
        del temp_dataset

    def _iter_datasets(self):
        if self.init_iterators:
            self.iterators = [iter(iterable) for iterable in self.iterables]
            self.init_iterators = False
        for weight in self.weights:
            self.index = (self.index + 1) % len(self.iterators)
            for i in range(weight):
                yield self.iterators[self.index]

    def _iter_examples(self):
        for iterator in cycle(self._iter_datasets()):
            yield next(iterator)

    def __iter__(self):
        while True:
            for minibatch in _pool(
                    self._iter_examples(),
                    self.batch_size,
                    self.batch_size_fn,
                    self.batch_size_multiple,
                    self.sort_key,
                    self.random_shuffler,
                    self.pool_factor):
                minibatch = sorted(minibatch, key=self.sort_key, reverse=True)
                yield torchtext.data.Batch(minibatch,
                                           self.iterables[0].dataset,
                                           self.device)


def build_dataset_iter_multiple(train_shards, fields, opt):
    return MultipleDatasetIterator(
        train_shards, fields, "cpu", opt)
