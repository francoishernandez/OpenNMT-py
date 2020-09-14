"""Transforms relate to tokenization/subword."""
from onmt.utils.logging import logger
from onmt.dynamic.transforms import register_transform
from .transform import Transform


_COMMON_OPT_ADDED = False  # only load common options one time


class TokenizerTransform(Transform):
    """Tokenizer transform abstract class."""

    def __init__(self, opts):
        """Initialize neccessary options for Tokenizer."""
        super().__init__(opts)
        self._parse_opts()

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to Subword."""
        global _COMMON_OPT_ADDED
        if _COMMON_OPT_ADDED is True:
            return
        group = parser.add_argument_group('Transform/Subword/Common')
        group.add('-src_subword_model', '--src_subword_model',
                  help="Path of subword model for src (or shared).")
        group.add('-tgt_subword_model', '--tgt_subword_model',
                  help="Path of subword model for tgt.")

        group.add('-subword_nbest', '--subword_nbest', type=int, default=1,
                  help="No of (n_best) candidate in subword regularization."
                  "Valid for unigram sampling, invalid for BPE-dropout.")
        group.add('-subword_alpha', '--subword_alpha', type=float, default=0,
                  help="Smoothing param for sentencepiece unigram sampling,"
                  "and dropout probability for BPE-dropout.")
        _COMMON_OPT_ADDED = True

    def _parse_opts(self):
        raise NotImplementedError

    def _set_subword_opts(self):
        """Set necessary options relate to subword."""
        self.share_vocab = self.opts.share_vocab
        self.src_subword_model = self.opts.src_subword_model
        self.tgt_subword_model = self.opts.tgt_subword_model
        self.subword_nbest = self.opts.subword_nbest
        self.alpha = self.opts.subword_alpha

    def __getstate__(self):
        """Pickling following for rebuild."""
        return self.opts

    def __setstate__(self, opts):
        """Reload when unpickling from save file."""
        self.opts = opts
        self._parse_opts()
        self.warm_up()


@register_transform(name='sentencepiece')
class SentencePieceTransform(TokenizerTransform):
    """SentencePiece subword transform class."""

    def __init__(self, opts):
        """Initialize neccessary options for sentencepiece."""
        super().__init__(opts)
        self._parse_opts()

    def _parse_opts(self):
        self._set_subword_opts()

    def warm_up(self, vocabs=None):
        """Load subword models."""
        import sentencepiece as spm
        load_src_model = spm.SentencePieceProcessor()
        load_src_model.Load(self.src_subword_model)
        if self.share_vocab:
            self.load_models = {
                'src': load_src_model,
                'tgt': load_src_model
            }
        else:
            load_tgt_model = spm.SentencePieceProcessor()
            load_tgt_model.Load(self.tgt_subword_model)
            self.load_models = {
                'src': load_src_model,
                'tgt': load_tgt_model
            }

    def _tokenize(self, tokens, side='src', is_train=False):
        """Do sentencepiece subword tokenize."""
        sp_model = self.load_models[side]
        sentence = ' '.join(tokens)
        if is_train is False or self.subword_nbest in [0, 1]:
            # derterministic subwording
            segmented = sp_model.encode(sentence, out_type=str)
        else:
            # subword sampling when subword_nbest > 1 or -1
            # alpha should be 0.0 < alpha < 1.0
            segmented = sp_model.encode(
                sentence, out_type=str, enable_sampling=True,
                alpha=self.alpha, nbest_size=self.subword_nbest)
        return segmented

    def apply(self, src, tgt, is_train=False, stats=None, **kwargs):
        """Apply sentencepiece subword encode to src & tgt."""
        src_out = self._tokenize(src, 'src', is_train)
        tgt_out = self._tokenize(tgt, 'tgt', is_train)
        if stats is not None:
            n_words = len(src) + len(tgt)
            n_subwords = len(src_out) + len(tgt_out)
            stats.subword(n_subwords, n_words)
        return src_out, tgt_out

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}, {}={}, {}={}, {}={}, {}={}'.format(
            'share_vocab', self.share_vocab,
            'subword_nbest', self.subword_nbest,
            'alpha', self.alpha,
            'src_subword_model', self.src_subword_model,
            'tgt_subword_model', self.tgt_subword_model
        )


@register_transform(name='bpe')
class BPETransform(TokenizerTransform):
    def __init__(self, opts):
        """Initialize neccessary options for subword_nmt."""
        super().__init__(opts)
        self._parse_opts()

    def _parse_opts(self):
        self._set_subword_opts()
        self.dropout = self.alpha

    def warm_up(self, vocabs=None):
        """Load subword models."""
        from subword_nmt.apply_bpe import BPE
        import codecs
        src_codes = codecs.open(self.src_subword_model, encoding='utf-8')
        load_src_model = BPE(codes=src_codes)
        if self.share_vocab:
            self.load_models = {
                'src': load_src_model,
                'tgt': load_src_model
            }
        else:
            tgt_codes = codecs.open(self.tgt_subword_model, encoding='utf-8')
            load_tgt_model = BPE(codes=tgt_codes)
            self.load_models = {
                'src': load_src_model,
                'tgt': load_tgt_model
            }

    def _tokenize(self, tokens, side='src', is_train=False):
        """Do bpe subword tokenize."""
        bpe_model = self.load_models[side]
        dropout = self.dropout if is_train else 0
        segmented = bpe_model.segment_tokens(tokens, dropout=dropout)
        return segmented

    def apply(self, src, tgt, is_train=False, stats=None, **kwargs):
        """Apply bpe subword encode to src & tgt."""
        src_out = self._tokenize(src, 'src', is_train)
        tgt_out = self._tokenize(tgt, 'tgt', is_train)
        if stats is not None:
            n_words = len(src) + len(tgt)
            n_subwords = len(src_out) + len(tgt_out)
            stats.subword(n_subwords, n_words)
        return src_out, tgt_out

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}, {}={}, {}={}, {}={}'.format(
            'share_vocab', self.share_vocab,
            'alpha', self.alpha,
            'src_subword_model', self.src_subword_model,
            'tgt_subword_model', self.tgt_subword_model
        )


@register_transform(name='onmt_tokenize')
class ONMTTokenizerTransform(TokenizerTransform):
    """OpenNMT Tokenizer transform class."""

    def __init__(self, opts):
        """Initialize neccessary options for OpenNMT Tokenizer."""
        super().__init__(opts)
        self._parse_opts()

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to Subword."""
        super().add_options(parser)
        group = parser.add_argument_group('Transform/Subword/ONMTTOK')
        group.add('-src_subword_type', '--src_subword_type',
                  type=str, default='none',
                  choices=['none', 'sentencepiece', 'bpe'],
                  help="Type of subword model for src (or shared) in onmttok.")
        group.add('-tgt_subword_type', '--tgt_subword_type',
                  type=str, default='none',
                  choices=['none', 'sentencepiece', 'bpe'],
                  help="Type of subword model for tgt in onmttok.")
        group.add('-onmttok_kwargs', '--onmttok_kwargs', type=str,
                  default="{'mode': 'none'}",
                  help="Accept any OpenNMT Tokenizer's options in dict string,"
                  "except subword related options listed earlier.")

    def _set_subword_opts(self):
        """Set all options relate to subword for OpenNMT/Tokenizer."""
        super()._set_subword_opts()
        self.src_subword_type = self.opts.src_subword_type
        self.tgt_subword_type = self.opts.tgt_subword_type

    def _parse_opts(self):
        self._set_subword_opts()
        logger.info("Parsed additional kwargs for OpenNMT Tokenizer {}".format(
            self.opts.onmttok_kwargs))
        self.other_kwargs = self.opts.onmttok_kwargs

    @classmethod
    def get_specials(cls, opts):
        src_specials, tgt_specials = set(), set()
        if opts.onmttok_kwargs.get("case_markup", False):
            _case_specials = ['｟mrk_case_modifier_C｠',
                              '｟mrk_begin_case_region_U｠',
                              '｟mrk_end_case_region_U｠']
            src_specials.update(_case_specials)
            tgt_specials.update(_case_specials)
        return (set(), set())

    def _get_subword_kwargs(self, side='src'):
        """Return a dict containing kwargs relate to `side` subwords."""
        subword_type = self.tgt_subword_type if side == 'tgt' \
            else self.src_subword_type
        subword_model = self.tgt_subword_model if side == 'tgt' \
            else self.src_subword_model
        kwopts = dict()
        if subword_type == 'bpe':
            kwopts['bpe_model_path'] = subword_model
            kwopts['bpe_dropout'] = self.alpha
        elif subword_type == 'sentencepiece':
            kwopts['sp_model_path'] = subword_model
            kwopts['sp_nbest_size'] = self.subword_nbest
            kwopts['sp_alpha'] = self.alpha
        else:
            raise ValueError(f'Unvalid subword_type: {subword_type}.')
        return kwopts

    def warm_up(self, vocab=None):
        """Initilize Tokenizer models."""
        import pyonmttok
        src_subword_kwargs = self._get_subword_kwargs(side='src')
        src_tokenizer = pyonmttok.Tokenizer(
            **src_subword_kwargs, **self.other_kwargs
        )
        if self.share_vocab:
            self.load_models = {
                'src': src_tokenizer,
                'tgt': src_tokenizer
            }
        else:
            tgt_subword_kwargs = self._get_subword_kwargs(side='tgt')
            tgt_tokenizer = pyonmttok.Tokenizer(
                **tgt_subword_kwargs, **self.other_kwargs
            )
            self.load_models = {
                'src': src_tokenizer,
                'tgt': tgt_tokenizer
            }

    def _tokenize(self, tokens, side='src', is_train=False):
        """Do OpenNMT Tokenizer's tokenize."""
        tokenizer = self.load_models[side]
        sentence = ' '.join(tokens)
        segmented, _ = tokenizer.tokenize(sentence)
        return segmented

    def apply(self, src, tgt, is_train=False, stats=None, **kwargs):
        """Apply OpenNMT Tokenizer to src & tgt."""
        src_out = self._tokenize(src, 'src')
        tgt_out = self._tokenize(tgt, 'tgt')
        if stats is not None:
            n_words = len(src) + len(tgt)
            n_subwords = len(src_out) + len(tgt_out)
            stats.subword(n_subwords, n_words)
        return src_out, tgt_out

    def _repr_args(self):
        """Return str represent key arguments for class."""
        repr_str = '{}={}'.format('share_vocab', self.share_vocab)
        for key, value in self.other_kwargs.items():
            repr_str += ', {}={}'.format(key, value)
        repr_str += ', src_subword_kwargs={}'.format(
            self._get_subword_kwargs(side='src'))
        repr_str += ', tgt_subword_kwargs={}'.format(
            self._get_subword_kwargs(side='tgt'))
        return repr_str
