"""Module for build dynamic fields."""
from collections import Counter, defaultdict
import os
import torch
from onmt.utils.logging import logger
from onmt.inputters.inputter import get_fields, _load_vocab, \
    _build_fields_vocab


def _get_dynamic_fields():
    # TODO: support for features & other opts
    src_nfeats = 0
    tgt_nfeats = 0
    fields = get_fields('text', src_nfeats, tgt_nfeats)

    # Do not support this in dynamic for now
    del fields['corpus_id']
    return fields


def build_dynamic_fields(opts, src_specials=None, tgt_specials=None):
    """Build fields for dynamic, including load & build vocab."""
    fields = _get_dynamic_fields()

    counters = defaultdict(Counter)
    logger.info("Loading vocab from text file...")

    _src_vocab, _src_vocab_size = _load_vocab(
        opts.src_vocab, 'src', counters,
        with_count=True)

    if opts.tgt_vocab:
        _tgt_vocab, _tgt_vocab_size = _load_vocab(
            opts.tgt_vocab, 'tgt', counters,
            with_count=True)
    elif opts.share_vocab:
        logger.info("Sharing src vocab to tgt...")
        counters['tgt'] = counters['src']
    else:
        raise ValueError("-tgt_vocab should be specified if not share_vocab.")

    logger.info("Building fields with vocab in counters...")
    fields = _build_fields_vocab(
        fields, counters, 'text', opts.share_vocab,
        opts.vocab_size_multiple,
        opts.src_vocab_size, opts.src_words_min_frequency,
        opts.tgt_vocab_size, opts.tgt_words_min_frequency,
        src_specials=src_specials, tgt_specials=tgt_specials)

    return fields


def get_vocabs(fields):
    """Get a dict contain src & tgt vocab list extracted from fields."""
    src_vocab = fields['src'].base_field.vocab.itos
    tgt_vocab = fields['tgt'].base_field.vocab.itos
    vocabs = {'src': src_vocab, 'tgt': tgt_vocab}
    return vocabs


def save_fields(opts, fields):
    """Dump `fields` object."""
    fields_path = "{}.vocab.pt".format(opts.save_data)
    os.makedirs(os.path.dirname(fields_path), exist_ok=True)
    logger.info(f"Saving fields to {fields_path}...")
    torch.save(fields, fields_path)


def load_fields(opts):
    """Load dumped `fields` object."""
    fields_path = "{}.vocab.pt".format(opts.save_data)
    logger.info(f"Loading fields from {fields_path}...")
    fields = torch.load(fields_path)
    return fields
