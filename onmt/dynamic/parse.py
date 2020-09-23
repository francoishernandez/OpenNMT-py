"""Parse cls for dynamic."""
import os
from onmt.utils.parse import ArgumentParser
from onmt.utils.logging import logger
from onmt.constants import CorpusName
from onmt.dynamic.transforms import AVAILABLE_TRANSFORMS


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
                if cname != CorpusName.VALID:
                    logger.warning(f"Corpus {cname}'s weight should be given."
                                   " We default it to 1 for you.")
                corpus['weight'] = 1
        logger.info(f"Parsed {len(corpora)} corpora from -data.")
        opt.data = corpora

    @classmethod
    def _validate_transforms_opts(cls, opt):
        """Check options used by transforms."""
        for name, transform_cls in AVAILABLE_TRANSFORMS.items():
            if name in opt._all_transform:
                transform_cls._validate_options(opt)

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
    def _validate_vocab_opts(cls, opt, build_vocab_only=False):
        """Check options relate to vocab."""
        if opt.src_vocab:
            cls._validate_file(opt.src_vocab, info='src vocab')
        if opt.tgt_vocab:
            cls._validate_file(opt.tgt_vocab, info='tgt vocab')

        if not build_vocab_only:
            # Check embeddings stuff
            if opt.both_embeddings is not None:
                assert (opt.src_embeddings is None
                        and opt.tgt_embeddings is None), \
                    "You don't need -src_embeddings or -tgt_embeddings \
                    if -both_embeddings is set."

            if any([opt.both_embeddings is not None,
                    opt.src_embeddings is not None,
                    opt.tgt_embeddings is not None]):
                assert opt.embeddings_type is not None, \
                    "You need to specify an -embedding_type!"

    @classmethod
    def validate_prepare_opts(cls, opt, build_vocab_only=False):
        """Validate all options relate to prepare (data/transform/vocab)."""
        cls._validate_data(opt)
        cls._get_all_transform(opt)
        cls._validate_transforms_opts(opt)
        cls._validate_vocab_opts(opt, build_vocab_only=build_vocab_only)
