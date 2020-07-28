"""All options for dynamic running. Should be given in a yaml file."""
from onmt.opts import _train_general_opts, config_opts, model_opts


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


def _dynamic_shard_opts(parser):
    """Options related to sharding."""
    group = parser.add_argument_group('sharding')
    group.add('-shard_size', '--shard_size', type=int, default=1000000,
              help="Divide corpus into smaller multiple shard files, "
                   "each shard will have opt.shard_size samples "
                   "except last shard. ")


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
              help="Global transform pipeline applying to data."
                   "Can be specified in each corpus.")
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

    group.add('-n_samples_subword', '--n_samples_subword', type=int, default=1,
              help="number of sample (n_best) generated in subword.")
    group.add('-theta_subword', '--theta_subword', type=float, default=0,
              help="theta argument for subword regularization.")
    # TODO
    group.add('-switchout_temperature', '--switchout_temperature',
              type=float, default=0,
              help="sampling temperature for switchout.")
    group.add('-tokendrop_temperature', '--tokendrop_temperature',
              type=float, default=0,
              help="sampling temperature for token deletion.")
    group.add('-tokenmask_temperature', '--tokenmask_temperature',
              type=float, default=0,
              help="sampling temperature for token masking.")
    group.add('--src_seq_length', '-src_seq_length', type=int, default=200,
              help="Maximum source sequence length")
    group.add('--tgt_seq_length', '-tgt_seq_length', type=int, default=200,
              help="Maximum target sequence length")


def dynamic_preprocess_shard_opts(parser):
    """All options used in sharding phrase of preprocess."""
    data_config_opts(parser)
    _dynamic_corpus_opts(parser)
    _dynamic_shard_opts(parser)


def dynamic_preprocess_vocab_opts(parser):
    """All options used in vocab phrase of preprocess."""
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
