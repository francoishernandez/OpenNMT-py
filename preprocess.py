#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""
import codecs
import glob
import sys
import gc
import torch
from collections import Counter, defaultdict

from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import _build_fields_vocab,\
                                    _load_vocab

from functools import partial
from multiprocessing.pool import Pool, ThreadPool


def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, "
                             "to avoid overwriting them!\n" % path)
            sys.exit(1)


def process_one_shard(corpus_params, params):
    corpus_type, fields, src_reader, tgt_reader, opt, existing_fields,\
        src_vocab, tgt_vocab, filter_pred, maybe_id = corpus_params
    i, (src_shard, tgt_shard) = params
    # create one counter per shard
    sub_sub_counter = defaultdict(Counter)
    assert len(src_shard) == len(tgt_shard)
    logger.info("Building shard %d." % i)
    dataset = inputters.Dataset(
        fields,
        readers=([src_reader, tgt_reader]
                 if tgt_reader else [src_reader]),
        data=([("src", src_shard), ("tgt", tgt_shard)]
              if tgt_reader else [("src", src_shard)]),
        dirs=([opt.src_dir, None]
              if tgt_reader else [opt.src_dir]),
        sort_key=inputters.str2sortkey[opt.data_type],
        filter_pred=filter_pred
    )
    if corpus_type == "train" and existing_fields is None:
        for ex in dataset.examples:
            for name, field in fields.items():
                try:
                    f_iter = iter(field)
                except TypeError:
                    f_iter = [(name, field)]
                    all_data = [getattr(ex, name, None)]
                else:
                    all_data = getattr(ex, name)
                for (sub_n, sub_f), fd in zip(
                        f_iter, all_data):
                    has_vocab = (sub_n == 'src' and
                                 src_vocab is not None) or \
                                (sub_n == 'tgt' and
                                 tgt_vocab is not None)
                    if (hasattr(sub_f, 'sequential')
                            and sub_f.sequential and not has_vocab):
                        val = fd
                        sub_sub_counter[sub_n].update(val)
    if maybe_id:
        shard_base = corpus_type + "_" + maybe_id
    else:
        shard_base = corpus_type
    data_path = "{:s}.{:s}.{:d}.pt".\
        format(opt.save_data, shard_base, i)

    logger.info(" * saving %sth %s data shard to %s."
                % (i, shard_base, data_path))

    dataset.save(data_path)

    del dataset.examples
    gc.collect()
    del dataset
    gc.collect()

    return sub_sub_counter


def build_save_one_corpus(dataset_params, params):
    corpus_type, fields, src_reader, tgt_reader,\
        opt, existing_fields, src_vocab, tgt_vocab = dataset_params
    src, tgt, maybe_id = params
    logger.info("Reading source and target files: %s %s." % (src, tgt))

    # create one counter per corpus
    sub_counter = defaultdict(Counter)

    src_shards = split_corpus(src, opt.shard_size)
    tgt_shards = split_corpus(tgt, opt.shard_size)
    shard_pairs = zip(src_shards, tgt_shards)

    if (corpus_type == "train" or opt.filter_valid) and tgt is not None:
        filter_pred = partial(
            inputters.filter_example, use_src_len=opt.data_type == "text",
            max_src_len=opt.src_seq_length, max_tgt_len=opt.tgt_seq_length)
    else:
        filter_pred = None

    with Pool(opt.shards_threads) as p:
        corpus_params = (process_one_shard, corpus_type, fields,
                         src_reader, tgt_reader, opt, existing_fields,
                         src_vocab, tgt_vocab, filter_pred, maybe_id)
        func = partial(process_one_shard, corpus_params)
        for sub_sub_counter in p.imap(func, enumerate(shard_pairs)):
            for key, value in sub_sub_counter.items():
                sub_counter[key].update(value)

    return sub_counter


def build_save_dataset(corpus_type, fields, src_reader, tgt_reader, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        counters = defaultdict(Counter)
        srcs = opt.train_src
        tgts = opt.train_tgt
        ids = opt.train_ids
    else:
        srcs = [opt.valid_src]
        tgts = [opt.valid_tgt]
        ids = [None]

    if corpus_type == "train":
        existing_fields = None
        if opt.src_vocab != "":
            try:
                logger.info("Using existing vocabulary...")
                existing_fields = torch.load(opt.src_vocab)
            except torch.serialization.pickle.UnpicklingError:
                logger.info("Building vocab from text file...")
                src_vocab, src_vocab_size = _load_vocab(
                    opt.src_vocab, "src", counters,
                    opt.src_words_min_frequency)
        else:
            src_vocab = None

        if opt.tgt_vocab != "":
            tgt_vocab, tgt_vocab_size = _load_vocab(
                opt.tgt_vocab, "tgt", counters,
                opt.tgt_words_min_frequency)
        else:
            tgt_vocab = None

    with ThreadPool(opt.corpus_threads) as p:
        dataset_params = (corpus_type, fields, src_reader, tgt_reader,
                          opt, existing_fields, src_vocab, tgt_vocab)
        func = partial(build_save_one_corpus, dataset_params)
        for sub_counter in p.imap(func, zip(srcs, tgts, ids)):
            for key, value in sub_counter.items():
                counters[key].update(value)

    if corpus_type == "train":
        vocab_path = opt.save_data + '.vocab.pt'
        if existing_fields is None:
            fields = _build_fields_vocab(
                fields, counters, opt.data_type,
                opt.share_vocab, opt.vocab_size_multiple,
                opt.src_vocab_size, opt.src_words_min_frequency,
                opt.tgt_vocab_size, opt.tgt_words_min_frequency)
        else:
            fields = existing_fields
        torch.save(fields, vocab_path)


def build_save_vocab(train_dataset, fields, opt):
    fields = inputters.build_vocab(
        train_dataset, fields, opt.data_type, opt.share_vocab,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.tgt_vocab, opt.tgt_vocab_size, opt.tgt_words_min_frequency,
        vocab_size_multiple=opt.vocab_size_multiple
    )
    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(fields, vocab_path)


def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split(None, 1)[0]
        return len(first_tok.split(u"￨")) - 1


def main(opt):
    ArgumentParser.validate_preprocess_args(opt)
    torch.manual_seed(opt.seed)
    if not(opt.overwrite):
        check_existing_pt_files(opt)

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    src_nfeats = 0
    tgt_nfeats = 0
    for src, tgt in zip(opt.train_src, opt.train_tgt):
        src_nfeats += count_features(src) if opt.data_type == 'text' \
            else 0
        tgt_nfeats += count_features(tgt)  # tgt always text so far
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(
        opt.data_type,
        src_nfeats,
        tgt_nfeats,
        dynamic_dict=opt.dynamic_dict,
        src_truncate=opt.src_seq_length_trunc,
        tgt_truncate=opt.tgt_seq_length_trunc)

    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    tgt_reader = inputters.str2reader["text"].from_opt(opt)

    logger.info("Building & saving training data...")
    build_save_dataset(
        'train', fields, src_reader, tgt_reader, opt)

    if opt.valid_src and opt.valid_tgt:
        logger.info("Building & saving validation data...")
        build_save_dataset('valid', fields, src_reader, tgt_reader, opt)


def _get_parser():
    parser = ArgumentParser(description='preprocess.py')

    opts.config_opts(parser)
    opts.preprocess_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
