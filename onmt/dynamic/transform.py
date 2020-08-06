"""Module for dynamic data transfrom."""
import os
import random
import numpy as np
import torch
from onmt.utils.logging import logger
from onmt.dynamic.vocab import get_vocabs


class Transform(object):
    """A Base class that every transform method should derived from."""
    def __init__(self, opts):
        self.opts = opts

    def warm_up(self, vocabs=None):
        pass

    @classmethod
    def get_specials(cls, opts):
        return (set(), set())

    def apply(self, src, tgt, stats=None, **kwargs):
        """Apply transform to `src` & `tgt`.

        Args:
            src (list): a list of str, representing tokens;
            tgt (list): a list of str, representing tokens;
            stats (TransformStatistics): a statistic object.
        """
        raise NotImplementedError

    def stats(self):
        """Return statistic message."""
        return ''

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return ''

    def __repr__(self):
        cls_name = type(self).__name__
        cls_args = self._repr_args()
        return '{}({})'.format(cls_name, cls_args)


class TokenizerTransform(Transform):
    """Tokenizer transform abstract class."""

    def __init__(self, opts):
        """Initialize neccessary options for Tokenizer."""
        super().__init__(opts)
        self._parse_opts()

    def _parse_opts(self):
        raise NotImplementedError

    def _set_subword_opts(self):
        """Set all options relate to subword."""
        self.share_vocab = self.opts.share_vocab
        self.src_subword_type = self.opts.src_subword_type
        self.tgt_subword_type = self.opts.tgt_subword_type
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


class SentencePieceTransform(TokenizerTransform):
    """SentencePiece subword transform class."""

    def __init__(self, opts):
        """Initialize neccessary options for sentencepiece."""
        super().__init__(opts)
        self._parse_opts()

    def _parse_opts(self):
        super()._set_subword_opts()

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

    def _tokenize(self, tokens, side='src'):
        """Do sentencepiece subword tokenize."""
        sp_model = self.load_models[side]
        sentence = ' '.join(tokens)
        if self.subword_nbest in [0, 1]:
            # derterministic subwording
            segmented = sp_model.encode(sentence, out_type=str)
        else:
            # subword sampling when subword_nbest > 1 or -1
            # alpha should be 0.0 < alpha < 1.0
            segmented = sp_model.encode(
                sentence, out_type=str, enable_sampling=True,
                alpha=self.alpha, nbest_size=self.subword_nbest)
        return segmented

    def apply(self, src, tgt, stats=None, **kwargs):
        """Apply sentencepiece subword encode to src & tgt."""
        src_out = self._tokenize(src, 'src')
        tgt_out = self._tokenize(tgt, 'tgt')
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


class ONMTTokenizerTransform(TokenizerTransform):
    """OpenNMT Tokenizer transform class."""

    def __init__(self, opts):
        """Initialize neccessary options for OpenNMT Tokenizer."""
        super().__init__(opts)
        self._parse_opts()

    def _parse_opts(self):
        super()._set_subword_opts()
        # Handle other kwargs
        kwargs_opts = self.opts.onmttok_kwargs
        kwargs_dict = eval(kwargs_opts)
        if not isinstance(kwargs_dict, dict):
            raise ValueError(
                f"-tok_kwargs is not a dict valid string:{kwargs_opts}.")
        logger.info("Parsed additional kwargs for OpenNMT Tokenizer {}".format(
            kwargs_dict))
        self.other_kwargs = kwargs_dict

    def _get_subword_kwargs(self, side='src'):
        """Return a dict containing kwargs relate to `side` subwords."""
        subword_type = self.tgt_subword_type if side == 'tgt' \
            else self.src_subword_type
        subword_model = self.tgt_subword_model if side == 'tgt' \
            else self.src_subword_model
        kwopts = dict()
        if subword_type == 'bpe':
            kwopts['bpe_model_path'] = subword_model
            logger.warning(
                "OpenNMT Tokenizer do not support BPE dropout with bpe.")
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

    def _tokenize(self, tokens, side='src'):
        """Do OpenNMT Tokenizer's tokenize."""
        tokenizer = self.load_models[side]
        sentence = ' '.join(tokens)
        segmented, _ = tokenizer.tokenize(sentence)
        return segmented

    def apply(self, src, tgt, stats=None, **kwargs):
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


class FilterTooLongTransform(Transform):
    """Filter out sentence that are too long."""

    def __init__(self, opts):
        super().__init__(opts)
        self.src_seq_length = opts.src_seq_length
        self.tgt_seq_length = opts.tgt_seq_length

    def apply(self, src, tgt, stats=None, **kwargs):
        """Return None if too long else return as is."""
        if len(src) > self.src_seq_length or len(tgt) > self.tgt_seq_length:
            if stats is not None:
                stats.filter_too_long()
            return None
        else:
            return src, tgt

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}, {}={}'.format(
            'src_seq_length', self.src_seq_length,
            'tgt_seq_length', self.tgt_seq_length
        )


class HammingDistanceSampling(object):
    """Functions related to (negative) Hamming Distance Sampling."""

    def _softmax(self, x):
        softmax = np.exp(x)/sum(np.exp(x))
        return softmax

    def _sample_replace(self, vocab, reject):
        """Sample a token from `vocab` other than `reject`."""
        token = reject
        while token == reject:
            token = random.choice(vocab)
        return token

    def _sample_distance(self, tokens, temperature):
        """Sample number of tokens to corrupt from `tokens`."""
        n_tokens = len(tokens)
        indices = np.arange(n_tokens)
        logits = indices * -1 * temperature
        probs = self._softmax(logits)
        distance = np.random.choice(indices, p=probs)
        return distance

    def _sample_position(self, tokens, distance):
        n_tokens = len(tokens)
        chosen_indices = random.sample(range(n_tokens), k=distance)
        return chosen_indices


class SwitchOutTransform(Transform, HammingDistanceSampling):
    """SwitchOut."""

    def __init__(self, opts):
        super().__init__(opts)

    def warm_up(self, vocabs):
        self.vocab = vocabs
        self.temperature = self.opts.switchout_temperature

    def _switchout(self, tokens, vocab, stats=None):
        # 1. sample number of tokens to corrupt
        n_chosen = self._sample_distance(tokens, self.temperature)
        # 2. sample positions to corrput
        chosen_indices = self._sample_position(tokens, distance=n_chosen)
        # 3. sample corrupted values
        out = []
        for (i, tok) in enumerate(tokens):
            if i in chosen_indices:
                tok = self._sample_replace(vocab, reject=tok)
                out.append(tok)
            else:
                out.append(tok)
        if stats is not None:
            stats.switchout(n_switchout=n_chosen, n_total=len(tokens))
        return out

    def apply(self, src, tgt, stats=None, **kwargs):
        """Apply switchout to both src and tgt side tokens."""
        src = self._switchout(src, self.vocab['src'], stats)
        tgt = self._switchout(tgt, self.vocab['tgt'], stats)
        return src, tgt

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('switchout_temperature', self.temperature)


class TokenDropTransform(Transform, HammingDistanceSampling):
    """Random drop tokens from sentence."""

    def __init__(self, opts):
        super().__init__(opts)
        self.temperature = self.opts.tokendrop_temperature

    def _token_drop(self, tokens, stats=None):
        # 1. sample number of tokens to corrupt
        n_chosen = self._sample_distance(tokens, self.temperature)
        # 2. sample positions to corrput
        chosen_indices = self._sample_position(tokens, distance=n_chosen)
        # 3. Drop token on chosen position
        out = [tok for (i, tok) in enumerate(tokens)
               if i not in chosen_indices]
        if stats is not None:
            stats.token_drop(n_dropped=n_chosen, n_total=len(tokens))
        return out

    def apply(self, src, tgt, stats=None, **kwargs):
        """Apply token drop to both src and tgt side tokens."""
        src = self._token_drop(src, stats)
        tgt = self._token_drop(tgt, stats)
        return src, tgt

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('worddrop_temperature', self.temperature)


class TokenMaskTransform(Transform, HammingDistanceSampling):
    """Random mask tokens from src sentence."""

    MASK_TOK = '｟MASK｠'

    def __init__(self, opts):
        super().__init__(opts)
        self.temperature = opts.tokenmask_temperature

    @classmethod
    def get_specials(cls, opts):
        """Get special vocabs added by prefix transform."""
        return ({cls.MASK_TOK}, set())

    def _token_mask(self, tokens, stats=None):
        # 1. sample number of tokens to corrupt
        n_chosen = self._sample_distance(tokens, self.temperature)
        # 2. sample positions to corrput
        chosen_indices = self._sample_position(tokens, distance=n_chosen)
        # 3. mask word on chosen position
        out = []
        for (i, tok) in enumerate(tokens):
            tok = self.MASK_TOK if i in chosen_indices else tok
            out.append(tok)
        if stats is not None:
            stats.token_mask(n_masked=n_chosen, n_total=len(tokens))
        return out

    def apply(self, src, tgt, stats=None, **kwargs):
        """Apply word drop to both src and tgt side tokens."""
        src = self._token_mask(src, stats)
        return src, tgt

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('tokenmask_temperature', self.temperature)


class PrefixSrcTransform(Transform):
    """Add target language Prefix to src sentence."""

    def __init__(self, opts):
        super().__init__(opts)
        self.prefix_dict = self.get_prefix_dict(self.opts)

    @classmethod
    def _get_prefix(cls, corpus):
        """Get prefix string of a `corpus`."""
        prefix_tag_format = "｟_tgt_is_{}｠"
        if 'prefix' in corpus['transforms']:
            tl = corpus['tgt_lang']
            prefix = prefix_tag_format.format(tl)
        else:
            prefix = None
        return prefix

    @classmethod
    def get_prefix_dict(cls, opts):
        """Get all needed prefix correspond to corpus in `opts`."""
        prefix_dict = {}
        for c_name, corpus in opts.data.items():
            prefix = cls._get_prefix(corpus)
            if prefix is not None:
                logger.info(f"Get prefix for {c_name}: {prefix}")
                prefix_dict[c_name] = prefix
        return prefix_dict

    @classmethod
    def get_specials(cls, opts):
        """Get special vocabs added by prefix transform."""
        prefix_dict = cls.get_prefix_dict(opts)
        src_specials = set(prefix_dict.values())
        return (src_specials, set())

    def _prepend(self, tokens, corpus_name):
        """Get prefix according to `corpus_name` and prepend to `tokens`."""
        corpus_prefix = self.prefix_dict.get(corpus_name, None)
        if corpus_prefix is None:
            raise ValueError('corpus_name does not exist.')
        return [corpus_prefix] + tokens

    def apply(self, src, tgt, stats=None, **kwargs):
        """Prepend prefix to src side tokens."""
        corpus_name = kwargs.get('corpus_name', None)
        if corpus_name is None:
            raise ValueError('corpus_name is required.')
        src = self._prepend(src, corpus_name)
        return src, tgt

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return '{}={}'.format('prefix_dict', self.prefix_dict)


AVAILABLE_TRANSFORMS = {
    'sentencepiece': SentencePieceTransform,
    'onmt_tokenize': ONMTTokenizerTransform,
    'filtertoolong': FilterTooLongTransform,
    'switchout': SwitchOutTransform,
    'tokendrop': TokenDropTransform,
    'tokenmask': TokenMaskTransform,
    'prefix': PrefixSrcTransform
}


class TransformStatistics(object):
    """Return a statistic counter for Transform."""

    def __init__(self):
        """Initialize statistic counter."""
        self.reset()

    def reset(self):
        """Statistic counters for all transforms."""
        self.filtered = 0
        self.words, self.subwords = 0, 0
        self.n_switchouted, self.so_total = 0, 0
        self.n_dropped, self.td_total = 0, 0
        self.n_masked, self.tm_total = 0, 0

    def filter_too_long(self):
        """Update filtered sentence counter."""
        self.filtered += 1

    def subword(self, subwords, words):
        """Update subword counter."""
        self.words += words
        self.subwords += subwords

    def switchout(self, n_switchout, n_total):
        """Update switchout counter."""
        self.n_switchouted += n_switchout
        self.so_total += n_total

    def token_drop(self, n_dropped, n_total):
        """Update token drop counter."""
        self.n_dropped += n_dropped
        self.td_total += n_total

    def token_mask(self, n_masked, n_total):
        """Update token mask counter."""
        self.n_masked += n_masked
        self.tm_total += n_total

    def report(self):
        """Return transform statistics report and reset counter."""
        msg = ''
        if self.filtered > 0:
            msg += f'Filtred sentence: {self.filtered} sent\n'.format()
        if self.words > 0:
            msg += f'Subword(SP/Tokenizer): {self.words} -> {self.subwords} tok\n'  # noqa: E501
        if self.so_total > 0:
            msg += f'SwitchOut: {self.n_switchouted}/{self.so_total} tok\n'
        if self.td_total > 0:
            msg += f'Token dropped: {self.n_dropped}/{self.td_total} tok\n'
        if self.tm_total > 0:
            msg += f'Token masked: {self.n_masked}/{self.tm_total} tok\n'
        self.reset()
        return msg


class TransformPipe(Transform):
    """Pipeline built by a list of Transform instance."""

    def __init__(self, opts, transform_list):
        """Initialize pipeline by a list of transform instance."""
        self.opts = None  # opts is not required
        self.transforms = transform_list
        self.statistics = TransformStatistics()

    @classmethod
    def build_from(cls, transform_list):
        """Return a `TransformPipe` instance build from `transform_list`."""
        for transform in transform_list:
            assert isinstance(transform, Transform), \
                "transform should be a instance of Transform."
        transform_pipe = cls(None, transform_list)
        return transform_pipe

    def warm_up(self, vocabs):
        """Warm up Pipeline by iterate over all transfroms."""
        for transform in self.transforms:
            transform.warm_up(vocabs)

    @classmethod
    def get_specials(cls, opts, transforms):
        """Return all specials introduced by `transforms`."""
        src_specials, tgt_specials = set(), set()
        for transform in transforms:
            _src_special, _tgt_special = transform.get_specials(transform.opts)
            src_specials.update(_src_special)
            tgt_specials.update(tgt_specials)
        return (src_specials, tgt_specials)

    def apply(self, src, tgt, **kwargs):
        """Apply transform pipe to `src` & `tgt`.

        Args:
            src (list): a list of str, representing tokens;
            tgt (list): a list of str, representing tokens;

        """
        item = (src, tgt)
        for transform in self.transforms:
            item = transform.apply(*item, stats=self.statistics, **kwargs)
            if item is None:
                break
        return item

    def stats(self):
        """Return statistic message."""
        return self.statistics.report()

    def _repr_args(self):
        """Return str represent key arguments for class."""
        info_args = []
        for transform in self.transforms:
            info_args.append(repr(transform))
        return ', '.join(info_args)


def make_transforms(opts, transforms_cls, fields):
    """Build transforms in `transforms_cls` with vocab of `fields`."""
    vocabs = get_vocabs(fields)
    transforms = {}
    for name, transform_cls in transforms_cls.items():
        transform_obj = transform_cls(opts)
        transform_obj.warm_up(vocabs)
        transforms[name] = transform_obj
    return transforms


def get_transforms_cls(transform_names):
    """Return valid transform class indicated in `transform_names`."""
    transforms_cls = {}
    for name in transform_names:
        if name not in AVAILABLE_TRANSFORMS:
            raise ValueError("specified tranform not supported!")
        transforms_cls[name] = AVAILABLE_TRANSFORMS[name]
    return transforms_cls


def get_specials(opts, transforms_cls_dict):
    """Get specials of transforms that should be registed in Vocab."""
    all_specials = {'src': set(), 'tgt': set()}
    for name, transform_cls in transforms_cls_dict.items():
        src_specials, tgt_specials = transform_cls.get_specials(opts)
        all_specials['src'].update(src_specials)
        all_specials['tgt'].update(tgt_specials)
    logger.info(f"Get special vocabs from Transforms: {all_specials}.")
    return all_specials


def save_transforms(opts, transforms):
    """Dump `transforms` object."""
    transforms_path = "{}.transforms.pt".format(opts.save_data)
    os.makedirs(os.path.dirname(transforms_path), exist_ok=True)
    logger.info(f"Saving Transforms to {transforms_path}.")
    torch.save(transforms, transforms_path)


def load_transforms(opts):
    """Load dumped `transforms` object."""
    transforms_path = "{}.transforms.pt".format(opts.save_data)
    transforms = torch.load(transforms_path)
    logger.info("Transforms loaded.")
    return transforms
