#!/usr/bin/env python
"""Train models with dynamic data."""
import torch

# import onmt.opts as opts
from onmt.utils.distributed import ErrorHandler, consumer, batch_producer
from onmt.utils.misc import set_random_seed, read_embeddings,\
    calc_vocab_load_stats, convert_to_torch_tensor
from onmt.utils.logging import init_logger, logger

from onmt.train_single import main as single_main, get_train_iter

from onmt.dynamic.parse import DynamicArgumentParser
from onmt.dynamic.opts import dynamic_train_opts
from onmt.dynamic.corpus import save_transformed_sample
from onmt.dynamic.vocab import build_dynamic_fields, save_fields
from onmt.dynamic.transforms import make_transforms, save_transforms, \
    get_specials, get_transforms_cls

# Set sharing strategy manually instead of default based on the OS.
torch.multiprocessing.set_sharing_strategy('file_system')


def prepare_embeddings(opt, fields):
    if any([opt.both_embeddings is not None,
            opt.src_embeddings is not None,
            opt.tgt_embeddings is not None]):
        vocs = []
        for side in ['src', 'tgt']:
            try:
                vocab = fields[side].base_field.vocab
            except AttributeError:
                vocab = fields[side].vocab
            vocs.append(vocab)
        enc_vocab, dec_vocab = vocs

    skip_lines = 1 if opt.embeddings_type == "word2vec" else 0
    if opt.both_embeddings is not None:
        set_of_src_and_tgt_vocab = \
            set(enc_vocab.stoi.keys()) | set(dec_vocab.stoi.keys())
        logger.info("Reading encoder and decoder embeddings from {}".format(
            opt.both_embeddings))
        src_vectors, total_vec_count = \
            read_embeddings(opt.both_embeddings, skip_lines,
                            set_of_src_and_tgt_vocab)
        tgt_vectors = src_vectors
        logger.info("\tFound {} total vectors in file".format(total_vec_count))
    else:
        if opt.src_embeddings is not None:
            logger.info("Reading encoder embeddings from {}".format(
                opt.src_embeddings))
            src_vectors, total_vec_count = read_embeddings(
                opt.src_embeddings, skip_lines,
                filter_set=enc_vocab.stoi
            )
            logger.info("\tFound {} total vectors in file.".format(
                total_vec_count))
        else:
            src_vectors = None
        if opt.tgt_embeddings is not None:
            logger.info("Reading decoder embeddings from {}".format(
                opt.tgt_embeddings))
            tgt_vectors, total_vec_count = read_embeddings(
                opt.tgt_embeddings, skip_lines,
                filter_set=dec_vocab.stoi
            )
            logger.info("\tFound {} total vectors in file".format(total_vec_count))
        else:
            tgt_vectors = None
    logger.info("After filtering to vectors in vocab:")
    if opt.src_embeddings is not None or opt.both_embeddings is not None:
        logger.info("\t* enc: %d match, %d missing, (%.2f%%)"
                    % calc_vocab_load_stats(enc_vocab, src_vectors))
    if opt.tgt_embeddings is not None or opt.both_embeddings is not None:
        logger.info("\t* dec: %d match, %d missing, (%.2f%%)"
                    % calc_vocab_load_stats(dec_vocab, tgt_vectors))

    # Write to file
    enc_output_file = opt.save_data + ".enc_embeddings.pt"
    dec_output_file = opt.save_data + ".dec_embeddings.pt"
    if opt.src_embeddings is not None or opt.both_embeddings is not None:
        logger.info("\nSaving encoder embeddings as:\n\t* enc: %s"
                    % enc_output_file)
        torch.save(
            convert_to_torch_tensor(src_vectors, enc_vocab),
            enc_output_file
        )
        # set the opt in place
        opt.pre_word_vecs_enc = enc_output_file
    if opt.tgt_embeddings is not None or opt.both_embeddings is not None:
        logger.info("\nSaving decoder embeddings as:\n\t* dec: %s"
                    % dec_output_file)
        torch.save(
            convert_to_torch_tensor(tgt_vectors, dec_vocab),
            dec_output_file
        )
        # set the opt in place
        opt.pre_word_vecs_dec = dec_output_file


def prepare_fields_transforms(opt):
    """Prepare or dump fields & transforms before training."""
    DynamicArgumentParser.validate_dynamic_corpus(opt)
    DynamicArgumentParser.get_all_transform(opt)

    transforms_cls = get_transforms_cls(opt._all_transform)
    specials = get_specials(opt, transforms_cls)

    fields = build_dynamic_fields(
        opt, src_specials=specials['src'], tgt_specials=specials['tgt'])
    save_fields(opt, fields)

    # maybe do stuff for pretrained embeddings
    prepare_embeddings(opt, fields)


    transforms = make_transforms(opt, transforms_cls, fields)
    save_transforms(opt, transforms)
    if opt.n_sample > 1:
        save_transformed_sample(opt, transforms, n_sample=opt.n_sample)


def train(opt):
    DynamicArgumentParser.validate_train_opts(opt)
    DynamicArgumentParser.update_model_opts(opt)
    DynamicArgumentParser.validate_model_opts(opt)

    init_logger(opt.log_file)
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
            train_iter = get_train_iter(
                opt, dynamic=True, stride=nb_gpu, offset=device_id)
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
        single_main(opt, 0, dynamic=True)
    else:   # case only CPU
        single_main(opt, -1, dynamic=True)


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
