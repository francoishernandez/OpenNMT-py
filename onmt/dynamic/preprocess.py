#!/usr/bin/env python
from onmt.dynamic.parse import DynamicArgumentParser
from onmt.dynamic.opts import dynamic_preprocess_shard_opts,\
    dynamic_preprocess_vocab_opts
from onmt.dynamic.shard import sharding
from onmt.dynamic.vocab import build_dynamic_fields, save_fields
from onmt.dynamic.transform import make_transforms, save_transforms, \
    get_specials, get_transforms_cls


def shard_main(opts):
    DynamicArgumentParser.valid_dynamic_corpus(opts)
    sharding(opts.data, opts.shard_size, opts.save_data)


def vocab_main(opts):
    DynamicArgumentParser.valid_dynamic_corpus(opts)
    DynamicArgumentParser.get_all_transform(opts)
    print(opts)
    transforms_cls = get_transforms_cls(opts._all_transform)
    specials = get_specials(opts, transforms_cls)

    fields = build_dynamic_fields(
        opts, src_specials=specials['src'], tgt_specials=specials['tgt'])
    save_fields(opts, fields)

    transfroms = make_transforms(opts, transforms_cls, fields)
    save_transforms(opts, transfroms)


def get_parser():
    parser = DynamicArgumentParser(description='dynamic_preprocess.py')
    subparsers = parser.add_subparsers(
        help="Choose subcommand to execute.", dest="subcommand")

    shard_parser = subparsers.add_parser(
        "shard", help="Build shards for parallel corpus.")
    dynamic_preprocess_shard_opts(shard_parser)

    vocab_parser = subparsers.add_parser(
        "vocab", help="Build fields & transform.")
    dynamic_preprocess_vocab_opts(vocab_parser)
    return parser


def main():
    parser = get_parser()
    opts, unknown = parser.parse_known_args()
    if opts.subcommand == 'shard':
        shard_main(opts)
    elif opts.subcommand == 'vocab':
        vocab_main(opts)


if __name__ == '__main__':
    main()
