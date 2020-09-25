"""All options for dynamic running. Should be given in a yaml file."""
from onmt.opts import _train_general_opts, config_opts, model_opts,\
    _add_reproducibility_opts
from onmt.dynamic.transforms import AVAILABLE_TRANSFORMS


def _dynamic_corpus_opts(parser, build_vocab_only=False):
    """Options related to training corpus, type: a list of dictionary."""
    group = parser.add_argument_group('Data')
    group.add("-data", "--data", required=True,
              help="List of datasets and their specifications. "
                   "See examples/*.yaml for further details.")
    group.add("-skip_empty_level", "--skip_empty_level", default="warning",
              choices=["silent", "warning", "error"],
              help="Security level when encounter empty examples."
                   "silent: silently ignore/skip empty example;"
                   "warning: warning when ignore/skip empty example;"
                   "error: raise error & stop excution when encouter empty.)")
    group.add("-transforms", "--transforms", default=[], nargs="+",
              choices=AVAILABLE_TRANSFORMS.keys(),
              help="Default transform pipeline to apply to data. "
                   "Can be specified in each corpus of data to override.")

    group.add("-save_data", "--save_data", required=build_vocab_only,
              help="Output base path for objects that will "
                   "be saved (vocab, transforms, embeddings, ...).")
    group.add("-overwrite", "--overwrite", action="store_true",
              help="Overwrite existing objects if any.")
    group.add(
        '-n_sample', '--n_sample',
        type=int, default=(5000 if build_vocab_only else 0),
        help=("Build vocab using " if build_vocab_only else "Stop after save ")
        + "this number of transformed samples/corpus. Can be [-1, 0, N>0]. "
        "Set to -1 to go full corpus, 0 to skip.")

    if not build_vocab_only:
        group.add('-dump_fields', '--dump_fields', action='store_true',
                  help="Dump fields `*.vocab.pt` to disk."
                  " -save_data should be set as saving prefix.")
        group.add('-dump_transforms', '--dump_transforms', action='store_true',
                  help="Dump transforms `*.transforms.pt` to disk."
                  " -save_data should be set as saving prefix.")


def _dynamic_fields_opts(parser, build_vocab_only=False):
    """Options related to vocabulary and fields.

    Add all options relate to vocabulary or fields to parser.
    If `build_vocab_only` set to True, do not contain fields
    related options which won't be used in `bin/build_vocab.py`.
    """
    group = parser.add_argument_group("Vocab")
    group.add("-src_vocab", "--src_vocab",
              required=not(build_vocab_only),
              help="Path to a vocabulary file for src."
                   "Format: one <word> or <word>\t<count> per line.")
    group.add("-tgt_vocab", "--tgt_vocab",
              help="Path to a vocabulary file for tgt."
                   "Format: one <word> or <word>\t<count> per line.")
    group.add("-share_vocab", "--share_vocab", action="store_true",
              help="Share source and target vocabulary.")

    if not build_vocab_only:
        group.add("-src_vocab_size", "--src_vocab_size",
                  type=int, default=50000,
                  help="Maximum size of the source vocabulary.")
        group.add("-tgt_vocab_size", "--tgt_vocab_size",
                  type=int, default=50000,
                  help="Maximum size of the target vocabulary")
        group.add("-vocab_size_multiple", "--vocab_size_multiple",
                  type=int, default=1,
                  help="Make the vocabulary size a multiple of this value.")

        group.add("-src_words_min_frequency", "--src_words_min_frequency",
                  type=int, default=0,
                  help="Discard source words with lower frequency.")
        group.add("-tgt_words_min_frequency", "--tgt_words_min_frequency",
                  type=int, default=0,
                  help="Discard target words with lower frequency.")

        # Truncation options, for text corpus
        group = parser.add_argument_group("Pruning")
        group.add("--src_seq_length_trunc", "-src_seq_length_trunc",
                  type=int, default=None,
                  help="Truncate source sequence length.")
        group.add("--tgt_seq_length_trunc", "-tgt_seq_length_trunc",
                  type=int, default=None,
                  help="Truncate target sequence length.")

        group = parser.add_argument_group('Embeddings')
        group.add('-both_embeddings', '--both_embeddings',
                  help="Path to the embeddings file to use "
                  "for both source and target tokens.")
        group.add('-src_embeddings', '--src_embeddings',
                  help="Path to the embeddings file to use for source tokens.")
        group.add('-tgt_embeddings', '--tgt_embeddings',
                  help="Path to the embeddings file to use for target tokens.")
        group.add('-embeddings_type', '--embeddings_type',
                  choices=["GloVe", "word2vec"],
                  help="Type of embeddings file.")


def _dynamic_transform_opts(parser):
    """Options related to transforms.

    Options that specified in the definitions of each transform class
    at `onmt/dynamic/transforms/*.py`.
    """
    for name, transform_cls in AVAILABLE_TRANSFORMS.items():
        transform_cls.add_options(parser)


def dynamic_prepare_opts(parser, build_vocab_only=False):
    """Options related to data prepare in dynamic mode.

    Add all dynamic data prepare related options to parser.
    If `build_vocab_only` set to True, then only contains options that
    will be used in `onmt/bin/build_vocab.py`.
    """
    config_opts(parser)
    _dynamic_corpus_opts(parser, build_vocab_only=build_vocab_only)
    _dynamic_fields_opts(parser, build_vocab_only=build_vocab_only)
    _dynamic_transform_opts(parser)

    if build_vocab_only:
        _add_reproducibility_opts(parser)
        # as for False, this will be added in _train_general_opts


def _train_dynamic_data(parser):
    group = parser.add_argument_group("Dynamic data")
    group.add("-bucket_size", "--bucket_size", type=int, default=2048,
              help="Examples per dynamically generated torchtext Dataset.")


def dynamic_train_opts(parser):
    """All options used in train."""
    # options relate to data preprare
    dynamic_prepare_opts(parser, build_vocab_only=False)
    # options relate to train
    _train_dynamic_data(parser)
    model_opts(parser)
    _train_general_opts(parser)
