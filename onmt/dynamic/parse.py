"""Parse cls for dynamic."""
import os
from onmt.utils.parse import ArgumentParser
from onmt.utils.logging import logger


class DynamicArgumentParser(ArgumentParser):

    @staticmethod
    def _validate_file(file_path, info):
        """Check `file_path` is valid or raise `IOError`."""
        if not os.path.isfile(file_path):
            raise IOError(f"Please check path of your {info} file!")

    @classmethod
    def _validate_data(cls, opt):
        """Parse corpora specified in data field of YAML file."""
        import yaml
        default_transforms = opt.transforms
        if len(default_transforms) != 0:
            logger.info(f"Default transforms: {default_transforms}.")
        corpora = yaml.safe_load(opt.data)

        for cname, corpus in corpora.items():
            # Check Transforms
            _transforms = corpus.get('transforms', None)
            if _transforms is None:
                logger.info(f"Missing transforms field for {cname} data, "
                            f"set to default: {default_transforms}.")
                corpus['transforms'] = default_transforms
            # Check path
            path_src = corpus.get('path_src', None)
            path_tgt = corpus.get('path_tgt', None)
            if path_src is None or path_tgt is None:
                raise ValueError(f'Corpus {cname} path are required')
            else:
                cls._validate_file(path_src, info=f'{cname}/path_src')
                cls._validate_file(path_tgt, info=f'{cname}/path_tgt')
            path_align = corpus.get('path_align', None)
            if path_align is None:
                if hasattr(opt, 'lambda_align') and opt.lambda_align > 0.0:
                    raise ValueError(f'Corpus {cname} alignment file path are '
                                     'required when lambda_align > 0.0')
                corpus['path_align'] = None
            else:
                cls._validate_file(path_align, info=f'{cname}/path_align')
            # Check prefix: will be used when use prefix transform
            src_prefix = corpus.get('src_prefix', None)
            tgt_prefix = corpus.get('tgt_prefix', None)
            if src_prefix is None or tgt_prefix is None:
                if 'prefix' in corpus['transforms']:
                    raise ValueError(f'Corpus {cname} prefix are required.')
            # Check weight
            weight = corpus.get('weight', None)
            if weight is None:
                logger.warning(f"Corpus {cname}'s weight should be given."
                               " We default it to 1 for you.")
                corpus['weight'] = 1
        logger.info(f"Parsed {len(corpora)} corpora from -data.")
        opt.data = corpora

    @classmethod
    def _validate_transforms_opts(cls, opt):
        """Check options relate to transforms."""
        assert 0 <= opt.subword_alpha <= 1, \
            "subword_alpha should be in the range [0, 1]"
        kwargs_dict = eval(opt.onmttok_kwargs)
        if not isinstance(kwargs_dict, dict):
            raise ValueError(f"-onmttok_kwargs is not a dict valid string.")
        opt.onmttok_kwargs = kwargs_dict

    @classmethod
    def _get_all_transform(cls, opt):
        """Should only called after `_validate_data`."""
        all_transforms = set(opt.transforms)
        for cname, corpus in opt.data.items():
            _transforms = set(corpus['transforms'])
            if len(_transforms) != 0:
                all_transforms.update(_transforms)
        if hasattr(opt, 'lambda_align') and opt.lambda_align > 0.0:
            if not all_transforms.isdisjoint(
                    {'sentencepiece', 'bpe', 'onmt_tokenize'}):
                raise ValueError('lambda_align is not compatible with'
                                 ' on-the-fly tokenization.')
            if not all_transforms.isdisjoint(
                    {'tokendrop', 'prefix', 'bart'}):
                raise ValueError('lambda_align is not compatible yet with'
                                 ' potentiel token deletion/addition.')
        opt._all_transform = all_transforms

    @classmethod
    def _validate_vocab_opts(cls, opt):
        """Check options relate to vocab."""
        if opt.src_vocab:
            cls._validate_file(opt.src_vocab, info='src vocab')
        if opt.tgt_vocab:
            cls._validate_file(opt.tgt_vocab, info='tgt vocab')

    @classmethod
    def validate_prepare_opts(cls, opt):
        """Validate all options relate to prepare (data/transform/vocab)."""
        cls._validate_data(opt)
        cls._validate_transforms_opts(opt)
        cls._get_all_transform(opt)
        cls._validate_vocab_opts(opt)
