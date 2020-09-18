"""All options for dynamic running. Should be given in a yaml file."""
from onmt.opts import _train_general_opts, config_opts, model_opts
from onmt.dynamic.transforms import AVAILABLE_TRANSFORMS


def data_config_opts(parser):
    """Options related to dynamic data config file."""
    parser.add('-data_config', '--data_config', required=True,
               is_config_file_arg=True, help='data config file path')


def _dynamic_corpus_opts(parser):
    """Options related to training corpus, type: a list of dictionary."""
    group = parser.add_argument_group('Data')
    group.add('-data', '--data', required=True,
              help='data configuration list.')
    group.add('-save_data', '--save_data', required=True,
              help='Output directory for data saving path.')
    group.add('-overwrite', '--overwrite', action="store_true",
              help="Overwrite existing shards if any.")
    group.add('-transforms', '--transforms', default=[], nargs='+',
              choices=AVAILABLE_TRANSFORMS.keys(),
              help="Default transform pipeline applying to data."
                   "Can be specified in each corpus of data to override.")


def _dynamic_vocab_opts(parser):
    """Options related to vocabulary and fields."""
    group = parser.add_argument_group('vocab')
    group.add('-src_vocab', '--src_vocab', required=True,
              help="Path to an vocabulary file for src(or shard)."
                   "Format: one <word> or <word>\t<count> per line.")
    group.add('-tgt_vocab', '--tgt_vocab',
              help="Path to an vocabulary file for tgt."
                   "Format: one <word> or <word>\t<count> per line.")

    group.add('-src_vocab_size', '--src_vocab_size', type=int, default=50000,
              help="Size of the source vocabulary")
    group.add('-tgt_vocab_size', '--tgt_vocab_size', type=int, default=50000,
              help="Size of the target vocabulary")
    group.add('-vocab_size_multiple', '--vocab_size_multiple',
              type=int, default=1,
              help="Make the vocabulary size a multiple of this value")

    group.add('-src_words_min_frequency', '--src_words_min_frequency',
              type=int, default=0)
    group.add('-tgt_words_min_frequency', '--tgt_words_min_frequency',
              type=int, default=0)

    group.add('-dynamic_dict', '--dynamic_dict', action='store_true',
              help="Create dynamic dictionaries")
    group.add('-share_vocab', '--share_vocab', action='store_true',
              help="Share source and target vocabulary")

    # Truncation options, for text corpus
    group = parser.add_argument_group('Pruning')
    group.add('--src_seq_length_trunc', '-src_seq_length_trunc',
              type=int, default=None,
              help="Truncate source sequence length.")
    group.add('--tgt_seq_length_trunc', '-tgt_seq_length_trunc',
              type=int, default=None,
              help="Truncate target sequence length.")


def _dynamic_transform_opts(parser):
    """Options related to transforms."""
    for name, transform_cls in AVAILABLE_TRANSFORMS.items():
        transform_cls.add_options(parser)


def dynamic_prepare_opts(parser):
    """Options related to data prepare in dynamic mode."""
    data_config_opts(parser)
    _dynamic_corpus_opts(parser)
    _dynamic_vocab_opts(parser)
    _dynamic_transform_opts(parser)
    parser.add_argument(
        '-n_sample', '--n_sample', type=int, default=-1,
        help="Show this number of transformed samples from each corpus.")


def _train_dynamic_data(parser):
    group = parser.add_argument_group('Dynamic data')
    group.add('-bucket_size', '--bucket_size', type=int, default=2048,
              help='Examples per dynamically generated torchtext Dataset')


def dynamic_train_opts(parser):
    """All options used in train."""
    # options relate to data preprare
    dynamic_prepare_opts(parser)
    # options relate to train
    config_opts(parser)
    _train_dynamic_data(parser)
    model_opts(parser)
    _train_general_opts(parser)
