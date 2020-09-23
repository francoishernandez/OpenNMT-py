#!/usr/bin/env python
"""Train models with dynamic data."""
import sys
import torch

# import onmt.opts as opts
from onmt.utils.distributed import ErrorHandler, consumer, batch_producer
from onmt.utils.misc import set_random_seed
from onmt.utils.logging import logger

from onmt.train_single import main as single_main, get_train_iter

from onmt.dynamic.parse import DynamicArgumentParser
from onmt.dynamic.opts import dynamic_train_opts
from onmt.dynamic.corpus import save_transformed_sample
from onmt.dynamic.fields import build_dynamic_fields, save_fields
from onmt.dynamic.transforms import make_transforms, save_transforms, \
    get_specials, get_transforms_cls

# Set sharing strategy manually instead of default based on the OS.
torch.multiprocessing.set_sharing_strategy('file_system')


def prepare_fields_transforms(opt):
    """Prepare or dump fields & transforms before training."""
    DynamicArgumentParser.validate_prepare_opts(opt)

    transforms_cls = get_transforms_cls(opt._all_transform)
    specials = get_specials(opt, transforms_cls)

    fields = build_dynamic_fields(
        opt, src_specials=specials['src'], tgt_specials=specials['tgt'])
    save_fields(opt, fields)

    transforms = make_transforms(opt, transforms_cls, fields)
    save_transforms(opt, transforms)
    if opt.n_sample != 0:
        logger.warning(
            "`-n_sample` != 0: Training will not be started. "
            f"Stop after saving {opt.n_sample} samples/corpus.")
        save_transformed_sample(opt, transforms, n_sample=opt.n_sample)
        sys.exit("Sample saved, please check it before restart training.")


def train(opt):
    DynamicArgumentParser.validate_train_opts(opt)
    DynamicArgumentParser.update_model_opts(opt)
    DynamicArgumentParser.validate_model_opts(opt)

    set_random_seed(opt.seed, False)

    if not opt.train_from:
        prepare_fields_transforms(opt)

    nb_gpu = len(opt.gpu_ranks)

    if opt.world_size > 1:

        queues = []
        mp = torch.multiprocessing.get_context('spawn')
        semaphore = mp.Semaphore(opt.world_size * opt.queue_size)
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        for device_id in range(nb_gpu):
            q = mp.Queue(opt.queue_size)
            queues += [q]
            procs.append(mp.Process(target=consumer, args=(
                single_main, opt, device_id, error_queue, q, semaphore, True),
                daemon=True))
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        producers = []
        # This does not work if we merge with the first loop, not sure why
        for device_id in range(nb_gpu):
            # Get the iterator to generate from
            train_iter = get_train_iter(opt, stride=nb_gpu, offset=device_id)
            producer = mp.Process(target=batch_producer,
                                  args=(train_iter, queues[device_id],
                                        semaphore, opt,),
                                  daemon=True)
            producers.append(producer)
            producers[device_id].start()
            logger.info(" Starting producer process pid: {}  ".format(
                producers[device_id].pid))
            error_handler.add_child(producers[device_id].pid)

        for p in procs:
            p.join()
        # Once training is done, we can terminate the producers
        for p in producers:
            p.terminate()

    elif nb_gpu == 1:  # case 1 GPU only
        single_main(opt, 0)
    else:   # case only CPU
        single_main(opt, -1)


def _get_parser():
    parser = DynamicArgumentParser(description='dynamic_train.py')
    dynamic_train_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt, unknown = parser.parse_known_args()
    train(opt)


if __name__ == "__main__":
    main()
