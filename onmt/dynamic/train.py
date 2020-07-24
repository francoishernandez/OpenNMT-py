#!/usr/bin/env python
"""Train models with dynamic data."""
import torch

# import onmt.opts as opts
from onmt.utils.distributed import ErrorHandler, consumer  # batch_producer
from onmt.utils.misc import set_random_seed
from onmt.utils.logging import logger

from onmt.train_single import main as single_main, get_train_iter

from onmt.dynamic.parse import DynamicArgumentParser
from onmt.dynamic.opts import dynamic_train_opts

# Set sharing strategy manually instead of default based on the OS.
torch.multiprocessing.set_sharing_strategy('file_system')


def train(opt):
    DynamicArgumentParser.valid_dynamic_corpus(opt)

    DynamicArgumentParser.validate_train_opts(opt)
    DynamicArgumentParser.update_model_opts(opt)
    DynamicArgumentParser.validate_model_opts(opt)

    set_random_seed(opt.seed, False)

    nb_gpu = len(opt.gpu_ranks)

    if opt.world_size > 1:
        # Get the iterator to generate from
        # train_iter = get_train_iter(opt, dynamic=True)

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
        producer = mp.Process(target=batch_producer,
                              # args=(train_iter, queues, semaphore, opt,),
                              args=(queues, semaphore, opt,),
                              daemon=True)
        producer.start()
        error_handler.add_child(producer.pid)

        for p in procs:
            p.join()
        producer.terminate()

    elif nb_gpu == 1:  # case 1 GPU only
        single_main(opt, 0, dynamic=True)
    else:   # case only CPU
        single_main(opt, -1, dynamic=True)


def batch_producer(queues, semaphore, opt):
    """Produce batches to `queues`."""
    from itertools import cycle
    from onmt.utils.logging import init_logger

    init_logger(opt.log_file)
    set_random_seed(opt.seed, False)
    # generator_to_serve = iter(generator_to_serve)

    generator_to_serve = get_train_iter(opt, dynamic=True)

    def pred(x):
        """
        Filters batches that belong only
        to gpu_ranks of current node
        """
        for rank in opt.gpu_ranks:
            if x[0] % opt.world_size == rank:
                return True

    generator_to_serve = filter(
        pred, enumerate(generator_to_serve))

    def next_batch(device_id):
        new_batch = next(generator_to_serve)
        semaphore.acquire()
        return new_batch[1]

    b = next_batch(0)

    for device_id, q in cycle(enumerate(queues)):
        b.dataset = None
        # Move batch to correspond device_id
        # batch_to(b, device_id)

        # hack to dodge unpicklable `dict_keys`
        b.fields = list(b.fields)
        q.put(b)
        b = next_batch(device_id)


def _get_parser():
    parser = DynamicArgumentParser(description='dynamic_train.py')
    dynamic_train_opts(parser)
    # opts.config_opts(parser)
    # opts.model_opts(parser)
    # opts.train_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt, unknown = parser.parse_known_args()
    train(opt)


if __name__ == "__main__":
    main()
