"""All options for dynamic running. Should be given in a yaml file."""
from onmt.opts import _train_general_opts, config_opts, model_opts
from onmt.dynamic.transform import AVAILABLE_TRANSFORMS


def data_config_opts(parser):
    """Options related to dynamic data config file."""
    parser.add('-data_config', '--data_config', required=True,
               is_config_file_arg=True, help='data config file path')
    parser.add('-save_data_config', '--save_data_config', required=False,
               #is_write_out_config_file_arg=True,
               help='config file save path')


def _dynamic_corpus_opts(parser):
    """Options related to training corpus, type: a list of dictionary."""
    group = parser.add_argument_group('Data')
    group.add('-data', '--data', required=True,
              help='data configuration list.')
    group.add('-save_data', '--save_data', required=True,
              help='Output directory for data saving path.')
    group.add('--overwrite', '-overwrite', action="store_true",
              help="Overwrite existing shards if any.")


def _dynamic_vocab_opts(parser):
    """Options related to vocabulary."""
    group = parser.add_argument_group('vocab')
    group.add('-share_vocab', '--share_vocab', action='store_true',
              help="Share source and target vocabulary")

    group.add('-src_vocab', '--src_vocab', required=True,
              help="Path to an vocabulary file for src(or shard)."
                   "Format: one word per line.")
    group.add('-tgt_vocab', '--tgt_vocab',
              help="Path to an vocabulary file for tgt."
                   "Format: one word per line.")

    group.add('-src_vocab_size', '--src_vocab_size',
              type=int, default=50000,
              help="Size of the source vocabulary")
    group.add('-tgt_vocab_size', '--tgt_vocab_size',
              type=int, default=50000,
              help="Size of the target vocabulary")

    group.add('-src_words_min_frequency', '--src_words_min_frequency',
              type=int, default=0)
    group.add('-tgt_words_min_frequency', '--tgt_words_min_frequency',
              type=int, default=0)
    group.add('-vocab_size_multiple', '--vocab_size_multiple',
              type=int, default=1,
              help="Make the vocabulary size a multiple of this value")


def _dynamic_transform_opts(parser):
    """Options related to vocabulary."""
    group = parser.add_argument_group('transforms')
    group.add('-transforms', '--transforms', default=[], nargs='+',
              choices=AVAILABLE_TRANSFORMS.keys(),
              help="Default transform pipeline applying to data."
                   "Can be specified in each corpus to override.")
    # Subword
    group.add('-src_subword_model', '--src_subword_model',
              help="Path of subword model for src (or shared).")
    group.add('-src_subword_type', '--src_subword_type',
              type=str, default='none',
              choices=['none', 'sentencepiece', 'bpe'],
              help="Type of subword model for src (or shared).")

    group.add('-tgt_subword_model', '--tgt_subword_model',
              help="Path of subword model for tgt.")
    group.add('-tgt_subword_type', '--tgt_subword_type',
              type=str, default='none',
              choices=['none', 'sentencepiece', 'bpe'],
              help="Type of subword model for tgt.")

    group.add('-subword_nbest', '--subword_nbest', type=int, default=1,
              help="number of (n_best) candidate in subword regularization."
              "Valid for unigram sampling, invalid for BPE-dropout.")
    group.add('-subword_alpha', '--subword_alpha', type=float, default=0,
              help="Soothing parameter for sentencepiece unigram sampling,"
              "and merge probability for BPE-dropout.")
    group.add('-onmttok_kwargs', '--onmttok_kwargs', type=str,
              default="{'mode': 'none'}",
              help="Accept any OpenNMT Tokenizer's options in dict string,"
              "except subword related options listed earlier.")
    # TODO
    group.add('-switchout_temperature', '--switchout_temperature',
              type=float, default=1.0,
              help="sampling temperature for switchout. tau^(-1) in paper."
              "Smaller value makes data more diverse.")
    group.add('-tokendrop_temperature', '--tokendrop_temperature',
              type=float, default=1.0,
              help="sampling temperature for token deletion.")
    group.add('-tokenmask_temperature', '--tokenmask_temperature',
              type=float, default=1.0,
              help="sampling temperature for token masking.")
    group.add('--src_seq_length', '-src_seq_length', type=int, default=200,
              help="Maximum source sequence length")
    group.add('--tgt_seq_length', '-tgt_seq_length', type=int, default=200,
              help="Maximum target sequence length")


def dynamic_preprocess_opts(parser):
    """All options used in dynamic preprocess."""
    data_config_opts(parser)
    _dynamic_corpus_opts(parser)
    _dynamic_vocab_opts(parser)
    _dynamic_transform_opts(parser)


def _train_dynamic_data(parser):
    group = parser.add_argument_group('Dynamic data')
    group.add('--bucket_size', '-bucket_size', type=int, default=2048,
              help='Examples per dynamically generated torchtext Dataset')


def dynamic_train_opts(parser):
    """All options used in train."""
    data_config_opts(parser)
    config_opts(parser)
    _dynamic_corpus_opts(parser)
    _train_dynamic_data(parser)
    model_opts(parser)
    _train_general_opts(parser)
