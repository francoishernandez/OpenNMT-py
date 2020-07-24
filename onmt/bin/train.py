#!/usr/bin/env python
"""Train models."""
import torch

import onmt.opts as opts
from onmt.utils.distributed import ErrorHandler, consumer, batch_producer
from onmt.utils.misc import set_random_seed
from onmt.utils.logging import logger

from onmt.train_single import main as single_main, get_train_iter
from onmt.utils.parse import ArgumentParser


def train(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)

    set_random_seed(opt.seed, False)

    nb_gpu = len(opt.gpu_ranks)

    if opt.world_size > 1:
        # Get the iterator to generate from
        train_iter = get_train_iter(opt, dynamic=False)

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
                single_main, opt, device_id, error_queue, q, semaphore, False),
                daemon=True))
            procs[device_id].start()
            logger.info(" Starting process pid: %d  " % procs[device_id].pid)
            error_handler.add_child(procs[device_id].pid)
        producer = mp.Process(target=batch_producer,
                              args=(train_iter, queues, semaphore, opt,),
                              daemon=True)
        producer.start()
        error_handler.add_child(producer.pid)

        for p in procs:
            p.join()
        producer.terminate()

    elif nb_gpu == 1:  # case 1 GPU only
        single_main(opt, 0)
    else:   # case only CPU
        single_main(opt, -1)


def _get_parser():
    parser = ArgumentParser(description='train.py')

    opts.config_opts(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    train(opt)


if __name__ == "__main__":
    main()
