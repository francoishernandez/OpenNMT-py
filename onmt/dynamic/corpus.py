"""Module that contain shard utils for dynamic data."""
import os
from onmt.utils.logging import logger
from onmt.constants import CorpusName
from onmt.dynamic.transforms import TransformPipe

from collections import Counter


class File(object):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        if self.name is None:
            from itertools import repeat
            self._file = repeat(None)
        else:
            import codecs
            self._file = codecs.open(self.name, *self.args, **self.kwargs)
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.name is not None and self._file:
            self._file.close()


class ParallelCorpus(object):
    """A parallel corpus file pair that can be loaded to iterate."""

    def __init__(self, src, tgt, align=None):
        """Initialize src & tgt side file path."""
        self.src = src
        self.tgt = tgt
        self.align = align

    def load(self, offset=0, stride=1):
        """
        Load file and iterate by lines.
        `offset` and `stride` allow to iterate only on every
        `stride` example, starting from `offset`.
        """
        with File(self.src, mode='rb') as fs,\
                File(self.tgt, mode='rb') as ft,\
                File(self.align, mode='rb') as fa:
            logger.info(f"Loading {repr(self)}...")
            for i, (sline, tline, align) in enumerate(zip(fs, ft, fa)):
                if (i % stride) == offset:
                    sline = sline.decode('utf-8')
                    tline = tline.decode('utf-8')
                    example = {
                        'src': sline,
                        'tgt': tline
                    }
                    if align is not None:
                        example['align'] = align.decode('utf-8')
                    yield example

    def __repr__(self):
        cls_name = type(self).__name__
        return '{}({}, {}, align={})'.format(
            cls_name, self.src, self.tgt, self.align)


def get_corpora(opts, is_train=False):
    corpora_dict = {}
    if is_train:
        for corpus_id, corpus_dict in opts.data.items():
            if corpus_id != CorpusName.VALID:
                corpora_dict[corpus_id] = ParallelCorpus(
                    corpus_dict["path_src"],
                    corpus_dict["path_tgt"],
                    corpus_dict["path_align"])
    else:
        if CorpusName.VALID in opts.data.keys():
            corpora_dict[CorpusName.VALID] = ParallelCorpus(
                opts.data[CorpusName.VALID]["path_src"],
                opts.data[CorpusName.VALID]["path_tgt"],
                opts.data[CorpusName.VALID]["path_align"])
        else:
            return None
    return corpora_dict


class ParallelCorpusIterator(object):
    def __init__(self, cid, corpus, transform, infinitely=False,
                 stride=1, offset=0):
        self.cid = cid
        self.corpus = corpus
        self.transform = transform
        self.infinitely = infinitely
        self.stride = stride
        self.offset = offset

    def _tokenize(self, stream):
        for example in stream:
            src = example['src'].strip('\n').split()
            tgt = example['tgt'].strip('\n').split()
            example['src'], example['tgt'] = src, tgt
            if 'align' in example:
                example['align'] = example['align'].strip('\n').split()
            yield example

    def _transform(self, stream):
        for example in stream:
            # item = self.transform.apply(
            # example, is_train=self.infinitely, corpus_name=self.cid)
            item = (example, self.transform, self.cid)
            if item is not None:
                yield item
        report_msg = self.transform.stats()
        if report_msg != '':
            logger.info("Transform statistics for {}:\n{}".format(
                self.cid, report_msg))

    def _add_index(self, stream):
        for i, item in enumerate(stream):
            item[0]['indices'] = i * self.stride + self.offset
            yield item

    def _iter_corpus(self):
        corpus_stream = self.corpus.load(
            stride=self.stride, offset=self.offset)
        tokenized_corpus = self._tokenize(corpus_stream)
        transformed_corpus = self._transform(tokenized_corpus)
        indexed_corpus = self._add_index(transformed_corpus)
        yield from indexed_corpus

    def __iter__(self):
        if self.infinitely:
            while True:
                _iter = self._iter_corpus()
                yield from _iter
        else:
            yield from self._iter_corpus()


def build_corpora_iters(corpora, transforms, corpora_info, is_train=False,
                        stride=1, offset=0):
    """Return `ParallelCorpusIterator` for all corpora defined in opts."""
    corpora_iters = dict()
    for c_id, corpus in corpora.items():
        c_transform_names = corpora_info[c_id].get('transforms', [])
        corpus_transform = [transforms[name] for name in c_transform_names]
        transform_pipe = TransformPipe.build_from(corpus_transform)
        logger.info(f"{c_id}'s transforms: {str(transform_pipe)}")
        corpus_iter = ParallelCorpusIterator(
            c_id, corpus, transform_pipe, infinitely=is_train,
            stride=stride, offset=offset)
        corpora_iters[c_id] = corpus_iter
    return corpora_iters


def save_transformed_sample(opts, transforms, n_sample=3, build_vocab=False):
    """Save transformed data sample as specified in opts."""
    from onmt.dynamic.iterator import DatasetAdapter
    corpora = get_corpora(opts, is_train=True)
    if build_vocab:
        counter_src = Counter()
        counter_tgt = Counter()
    datasets_iterables = build_corpora_iters(
        corpora, transforms,
        opts.data, is_train=True)
    sample_path = os.path.join(
        os.path.dirname(opts.save_data), CorpusName.SAMPLE)
    os.makedirs(sample_path, exist_ok=True)
    for c_name, c_iter in datasets_iterables.items():
        dest_base = os.path.join(
            sample_path, "{}.{}".format(c_name, CorpusName.SAMPLE))
        with open(dest_base + ".src", 'w', encoding="utf-8") as f_src,\
                open(dest_base + ".tgt", 'w', encoding="utf-8") as f_tgt:
            for i, item in enumerate(c_iter):
                maybe_example = DatasetAdapter._process(item, is_train=True)
                if maybe_example is None:
                    continue
                src_line, tgt_line = maybe_example['src'], maybe_example['tgt']
                if build_vocab:
                    counter_src.update(src_line.split(' '))
                    counter_tgt.update(tgt_line.split(' '))
                f_src.write(src_line + '\n')
                f_tgt.write(tgt_line + '\n')
                if i >= n_sample:
                    break
    if build_vocab:
        return counter_src, counter_tgt
