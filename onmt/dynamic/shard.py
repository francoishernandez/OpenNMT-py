"""Module that contain shard utils for dynamic data."""
import os
import re
from collections import defaultdict
from itertools import cycle
from onmt.utils.misc import split_corpus
from onmt.utils.logging import logger


VALID_CORPUS_NAME = 'valid'


def save_file(lines, dest):
    """Save `lines` into `dest`."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, 'wb') as f:
        for line in lines:
            f.write(line)


def make_link(filepath, link):
    """Make a soft link with name `link` to `filepath`."""
    root = os.getcwd()
    filepath = os.path.join(root, filepath)
    link = os.path.join(root, link)
    os.symlink(filepath, link)


def sharding(corpora, shard_size, save_data):
    """Sharding `corpora` into smaller files to `save_data`.

    Args:
        corpora (list): a dictionary of corpus with name as keys;
        shard_size (int): max size for each shard;
        save_data (str): file prefix for saving shards.

    """
    os.makedirs(os.path.dirname(save_data), exist_ok=True)
    for corpus_name, corpus_dict in corpora.items():
        if corpus_name == VALID_CORPUS_NAME:
            dest_base = "{}.{}".format(save_data, VALID_CORPUS_NAME)
            make_link(corpus_dict['path_src'], dest_base + '.0.src')
            make_link(corpus_dict['path_tgt'], dest_base + '.0.tgt')
        else:
            dest_base = "{}.train_{}".format(save_data, corpus_name)
            src_shards = split_corpus(corpus_dict['path_src'], shard_size)
            tgt_shards = split_corpus(corpus_dict['path_tgt'], shard_size)
            for i, (s_s, t_s) in enumerate(zip(src_shards, tgt_shards)):
                dest_shard = "{}.{}".format(dest_base, i)
                save_file(s_s, dest_shard + '.src')
                save_file(t_s, dest_shard + '.tgt')


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
            slines = fs.readlines()
            tlines = ft.readlines()
        if len(slines) != len(tlines):
            raise ValueError("src & tgt should be same length!")
        for sline, tline in zip(slines, tlines):
            sline = sline.decode('utf-8')
            tline = tline.decode('utf-8')
            yield (sline, tline)

    def __repr__(self):
        cls_name = type(self).__name__
        return '{}({}, {})'.format(cls_name, self.src, self.tgt)


def get_named_shards(shards_dir, shards_prefix, corpus_name):
    """Get `corpus_name`'s shard files in `shards_dir`."""
    if corpus_name != VALID_CORPUS_NAME:
        shards_pattern = r"{}.train_{}.([0-9]*).(src|tgt)".format(
            shards_prefix, corpus_name)
    else:
        shards_pattern = r"{}.{}.([0-9]*).(src|tgt)".format(
            shards_prefix, corpus_name)
    shards_pattern_obj = re.compile(shards_pattern)

    shards = defaultdict(lambda: dict(src=None, tgt=None))
    for f in os.listdir(shards_dir):
        matched = shards_pattern_obj.match(f)
        if matched is not None:
            path = os.path.join(shards_dir, f)
            no = int(matched.group(1))
            side = matched.group(2)
            shards[no][side] = path

    # sanity check
    if len(shards) == 0:
        raise Exception("Shard {}: not found in {}.".format(
            corpus_name, shards_dir))
    for no, shard_dict in shards.items():
        for side, path in shard_dict.items():
            if path is None:
                raise Exception("Shard {}: Missing no.{} for side {}.".format(
                    corpus_name, no, side))

    valid_no = sorted(shards.keys())
    logger.info(f"Corpus {corpus_name}: got {len(valid_no)} Shards.")

    # Return a sorted list of ParallelShard rather than dictionary
    shards_list = [ParallelShard(**shards[no]) for no in valid_no]
    return shards_list


def get_corpora_shards(opts, is_train=False):
    """Return a dictionary contain all shards for every corpus of opts."""
    shards_dir = os.path.dirname(opts.save_data)
    shards_basename = os.path.basename(opts.save_data)

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

    shards_dict = {}
    for corpus_id in corpora_ids:
        shards = get_named_shards(
            shards_dir, shards_basename, corpus_id)
        shards_dict[corpus_id] = shards
    return shards_dict


class ShardedCorpusIterator(object):
    """Generate examples from Sharded corpus."""
    def __init__(self, shards, transforms, infinitely=False):
        """Initialize."""
        self.shards = shards
        self.infinitely = infinitely
        self.transforms = transforms

    def _tokenize(self, stream):
        for (sline, tline) in stream:
            sline = sline.strip('\n').split()
            tline = tline.strip('\n').split()
            yield (sline, tline)

    def _transform(self, stream):
        for item in stream:
            for transform in self.transforms:
                item = transform.apply(*item)
            yield item

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


def build_sharded_corpora_iters(corpora_shards, transforms, corpora_info, train=False):  # noqa: E501
    """Return `ShardedCorpusIterator` for all corpora defined in opts."""
    corpora_iters = dict()
    for c_id, corpus_shards in corpora_shards.items():
        c_transform_names = corpora_info[c_id].get('transforms', [])
        corpus_transform = [transforms[name] for name in c_transform_names]
        logger.info(f"{c_id}'s transforms: {corpus_transform}")
        corpus_iter = ShardedCorpusIterator(
            corpus_shards, corpus_transform, infinitely=train)
        corpora_iters[c_id] = corpus_iter
    return corpora_iters
