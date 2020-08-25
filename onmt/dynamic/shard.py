"""Module that contain shard utils for dynamic data."""
import os
import re
from collections import defaultdict
from itertools import cycle
from onmt.utils.logging import logger
from onmt.dynamic.transform import TransformPipe


VALID_CORPUS_NAME = 'valid'


def make_link(filepath, link, overwrite=False):
    """Make a soft link with name `link` to `filepath`."""
    root = os.getcwd()
    filepath = os.path.join(root, filepath)
    link = os.path.join(root, link)
    if not os.path.exists(filepath):
        raise OSError(f"File not found at {filepath}.")
    if os.path.exists(link):
        if not overwrite:
            logger.warning(f"Link {link} already exist. Skipping")
            return
        else:
            logger.warning(f"Link {link} exist. overwriting it.")
            os.remove(link)
    os.symlink(filepath, link)


def save_dataset(corpora, save_data, overwrite=False):
    """Save `corpora` to `save_data`.

    Args:
        corpora (list): a dictionary of corpus with name as keys;
        save_data (str): file prefix for saving;
        overwrite (bool): if overwrite existing corpus.
    """
    os.makedirs(os.path.dirname(save_data), exist_ok=True)
    for corpus_name, corpus_dict in corpora.items():
        if corpus_name == VALID_CORPUS_NAME:
            dest_base = "{}.{}".format(save_data, VALID_CORPUS_NAME)
        else:
            dest_base = "{}.train_{}".format(save_data, corpus_name)
        make_link(corpus_dict['path_src'], dest_base + '.src', overwrite)
        make_link(corpus_dict['path_tgt'], dest_base + '.tgt', overwrite)


class ParallelShard(object):
    """A parallel shard file pair that can be loaded to iterate."""

    def __init__(self, src, tgt):
        """Initilize src & tgt side file path."""
        self.src = src
        self.tgt = tgt

    def load(self):
        """Load file and iterate by lines."""
        import codecs
        with codecs.open(self.src, mode='rb') as fs,\
                codecs.open(self.tgt, mode='rb') as ft:
            logger.info(f"Loading {repr(self)}...")
            for sline, tline in zip(fs, ft):
                sline = sline.decode('utf-8')
                tline = tline.decode('utf-8')
                yield (sline, tline)

    def __repr__(self):
        cls_name = type(self).__name__
        return '{}({}, {})'.format(cls_name, self.src, self.tgt)

class ParallelCorpus(object):
    """A parallel corpus file pair that can be loaded to iterate."""

    def __init__(self, src, tgt):
        """Initilize src & tgt side file path."""
        self.src = src
        self.tgt = tgt

    def load(self):
        """Load file and iterate by lines."""
        import codecs
        with codecs.open(self.src, mode='rb') as fs,\
                codecs.open(self.tgt, mode='rb') as ft:
            logger.info(f"Loading {repr(self)}...")
            for sline, tline in zip(fs, ft):
                sline = sline.decode('utf-8')
                tline = tline.decode('utf-8')
                yield (sline, tline)

    def __repr__(self):
        cls_name = type(self).__name__
        return '{}({}, {})'.format(cls_name, self.src, self.tgt)


def get_corpus_paths(base_dir, data_basename, corpus_name):
    if corpus_name != VALID_CORPUS_NAME:
        src = os.path.join(
            base_dir,
            "{}.train_{}.src".format(
                data_basename, corpus_name))
        tgt = os.path.join(
            base_dir,
            "{}.train_{}.tgt".format(
                data_basename, corpus_name))
    else:
        src = os.path.join(
            base_dir,
            "{}.{}.src".format(
                data_basename, corpus_name))
        tgt = os.path.join(
            base_dir,
            "{}.{}.tgt".format(
                data_basename, corpus_name))
    return ParallelCorpus(src, tgt)



# def get_named_shards(shards_dir, shards_prefix, corpus_name):
#     """Get `corpus_name`'s shard files in `shards_dir`."""
#     if corpus_name != VALID_CORPUS_NAME:
#         shards_pattern = r"{}.train_{}.([0-9]*).(src|tgt)".format(
#             shards_prefix, corpus_name)
#     else:
#         shards_pattern = r"{}.{}.([0-9]*).(src|tgt)".format(
#             shards_prefix, corpus_name)
#     shards_pattern_obj = re.compile(shards_pattern)

#     shards = defaultdict(lambda: dict(src=None, tgt=None))
#     for f in os.listdir(shards_dir):
#         matched = shards_pattern_obj.match(f)
#         if matched is not None:
#             path = os.path.join(shards_dir, f)
#             no = int(matched.group(1))
#             side = matched.group(2)
#             shards[no][side] = path

#     # sanity check
#     if len(shards) == 0:
#         raise Exception("Shard {}: not found in {}.".format(
#             corpus_name, shards_dir))
#     for no, shard_dict in shards.items():
#         for side, path in shard_dict.items():
#             if path is None:
#                 raise Exception("Shard {}: Missing no.{} for side {}.".format(
#                     corpus_name, no, side))

#     valid_no = sorted(shards.keys())
#     logger.info(f"Corpus {corpus_name}: got {len(valid_no)} Shards.")

#     # Return a sorted list of ParallelShard rather than dictionary
#     shards_list = [ParallelShard(**shards[no]) for no in valid_no]
#     return shards_list


def get_corpora(opts, is_train=False):
    base_dir = os.path.dirname(opts.save_data)
    data_basename = os.path.basename(opts.save_data)
    if is_train:
        corpora_ids = [c_id for c_id in opts.data.keys()
                       if c_id != VALID_CORPUS_NAME]
        if len(corpora_ids) == 0:
            raise ValueError("Please specify training corpus!")
    else:
        if VALID_CORPUS_NAME in opts.data.keys():
            corpora_ids = [VALID_CORPUS_NAME]
        else:
            return None
    corpora_dict = {}
    for corpus_id in corpora_ids:
        corpus = get_corpus_paths(
            base_dir, data_basename, corpus_id)
        corpora_dict[corpus_id] = corpus
    return corpora_dict


# def get_corpora_shards(opts, is_train=False):
#     """Return a dictionary contain all shards for every corpus of opts."""
#     shards_dir = os.path.dirname(opts.save_data)
#     shards_basename = os.path.basename(opts.save_data)

#     if is_train:
#         corpora_ids = [c_id for c_id in opts.data.keys()
#                        if c_id != VALID_CORPUS_NAME]
#         if len(corpora_ids) == 0:
#             raise ValueError("Please specify training corpus!")
#     else:
#         if VALID_CORPUS_NAME in opts.data.keys():
#             corpora_ids = [VALID_CORPUS_NAME]
#         else:
#             return None

#     shards_dict = {}
#     for corpus_id in corpora_ids:
#         shards = get_named_shards(
#             shards_dir, shards_basename, corpus_id)
#         shards_dict[corpus_id] = shards
#     return shards_dict


class ParallelCorpusIterator(object):
    def __init__(self, cid, corpus, transform, infinitely=False):
        self.cid = cid
        self.corpus = corpus
        self.transform = transform
        self.infinitely = infinitely

    def _tokenize(self, stream):
        for (sline, tline) in stream:
            sline = sline.strip('\n').split()
            tline = tline.strip('\n').split()
            yield (sline, tline)

    def _transform(self, stream):
        for item in stream:
            item = self.transform.apply(*item, corpus_name=self.cid)
            if item is not None:
                yield item
        report_msg = self.transform.stats()
        if report_msg != '':
            logger.info("Transform statistics for {}:\n{}".format(
                self.cid, report_msg))

    def _add_index(self, stream):
        for i, item in enumerate(stream):
            yield (*item, i)

    def _iter_corpus(self):
        corpus_stream = self.corpus.load()
        tokenized_corpus = self._tokenize(corpus_stream)
        transformed_corpus = self._transform(tokenized_corpus)
        indexed_corpus = self._add_index(transformed_corpus)
        yield from indexed_corpus

    def __iter__(self):
        if self.infinitely:
            yield from cycle(self._iter_corpus())
        else:
            yield from self._iter_corpus()


class ShardedCorpusIterator(object):
    """Generate examples from Sharded corpus.

    Corpus file will be opened, every lines will be passed to a
    [tokenize -> transform -> indexlize] pipeline.

    Args:
        cid (str): a string representing corpus id;
        shards (list): a list containing paths of the corpus;
        transform (onmt.dynamic.Transform): transforms to be applied to corpus;
        infinitely (bool): loop over corpus only once if False.

    Yield:
        (tuple): corpus tokenized examples been dynamicly transformed

    """

    def __init__(self, cid, shards, transform, infinitely=False):
        """Initialize a dynamic corpus iterator."""
        self.cid = cid
        self.shards = shards
        self.infinitely = infinitely
        self.transform = transform

    def _tokenize(self, stream):
        for (sline, tline) in stream:
            sline = sline.strip('\n').split()
            tline = tline.strip('\n').split()
            yield (sline, tline)

    def _transform(self, stream):
        for item in stream:
            item = self.transform.apply(*item, corpus_name=self.cid)
            if item is not None:
                yield item
        report_msg = self.transform.stats()
        if report_msg != '':
            logger.info("Transform statistics for {}:\n{}".format(
                self.cid, report_msg))

    def _add_index(self, stream):
        for i, item in enumerate(stream):
            yield (*item, i)

    def _iter_shard(self, shard):
        """Generate transformed lines from `shard` with line index."""
        shard_stream = shard.load()
        tokenized_shard = self._tokenize(shard_stream)
        transformed_shard = self._transform(tokenized_shard)
        indexed_shard = self._add_index(transformed_shard)
        yield from indexed_shard

    def __iter__(self):
        paths = self.shards
        if self.infinitely:
            paths = cycle(paths)
        for path in paths:
            yield from self._iter_shard(path)


def build_corpora_iters(corpora, transforms, corpora_info, train=False):  # noqa: E501
    """Return `ShardedCorpusIterator` for all corpora defined in opts."""
    corpora_iters = dict()
    for c_id, corpus in corpora.items():
        c_transform_names = corpora_info[c_id].get('transforms', [])
        corpus_transform = [transforms[name] for name in c_transform_names]
        transform_pipe = TransformPipe.build_from(corpus_transform)
        logger.info(f"{c_id}'s transforms: {str(transform_pipe)}")
        corpus_iter = ParallelCorpusIterator(
            c_id, corpus, transform_pipe, infinitely=train)
        corpora_iters[c_id] = corpus_iter
    return corpora_iters


def save_transformed_sample(opts, transforms, n_sample=3):
    """Save transformed data sample as specified in opts."""
    corpora_shards = get_corpora_shards(opts, is_train=True)
    datasets_iterables = build_sharded_corpora_iters(
        corpora_shards, transforms,
        opts.data, train=True)
    sample_path = os.path.join(
        os.path.dirname(opts.save_data), 'sample')
    os.makedirs(sample_path, exist_ok=True)
    for c_name, c_iter in datasets_iterables.items():
        dest_base = os.path.join(sample_path, "{}.sample".format(c_name))
        with open(dest_base, 'w', encoding="utf-8") as f_sample:
            for i, example in enumerate(c_iter):
                item_list = []
                for item in example:
                    if isinstance(item, list):
                        item = ' '.join(item)
                    item_list.append(str(item))
                line = '\t'.join(item_list)
                f_sample.write(line + '\n')
                if i > n_sample:
                    break
