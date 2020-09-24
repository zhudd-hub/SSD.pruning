from model.prune import l1normPruner
import model.prune as prune
import os
import argparse
import torch
from torchvision import datasets, transforms
import torch.optim as optim
from os.path import join
import json
from model.mythop import clever_format, profile
import logging
import torch.distributed as dist

from model.engine.inference import do_evaluation
from model.config import cfg
from model.data.build import make_data_loader
from model.engine.trainer import do_train
from model.modeling.detector import build_detection_model
from model.solver.build import make_optimizer, make_lr_scheduler
from model.utils import dist_util, mkdir
from model.utils.checkpoint import CheckPointer
from model.utils.dist_util import synchronize
from model.utils.logger import setup_logger
from model.utils.misc import str2bool


def main():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=1, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=1, type=int, help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--use_tensorboard', default=True, type=str2bool)
    parser.add_argument('--pruner', default='SlimmingPruner', type=str,
                    choices=['AutoSlimPruner', 'SlimmingPruner', 'l1normPruner'],
                    help='architecture to use')
    parser.add_argument('--pruneratio', default=0.4, type=float,
                    help='architecture to use')
    parser.add_argument('--sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    ######
    ##  prune
    ###########
    model = build_detection_model(cfg)
    newmodel = build_detection_model(cfg)
    checkpointer = CheckPointer(model, save_dir=cfg.OUTPUT_DIR)
    _= checkpointer.load()
    
    model.eval()
    newmodel.eval()

    if args.pruner == 'l1normPruner':
        kwargs = {'pruneratio': args.pruneratio}
    elif args.pruner == 'SlimmingPruner':
        kwargs = {'pruneratio': args.pruneratio}
    elif args.pruner == 'AutoSlimPruner':
        kwargs = {'prunestep': 16, 'constrain': 200e6}
    pruner = prune.__dict__[args.pruner](model=model, newmodel=newmodel, args=args, **kwargs)
    
    pruner.prune()
    ##---------count op
    input = torch.randn(1, 3, 320, 320)

    flops, params = profile(model, inputs=(input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    flopsnew, paramsnew = profile(newmodel, inputs=(input,), verbose=False)
    flopsnew, paramsnew = clever_format([flopsnew, paramsnew], "%.3f")
    logger.info("flops:{}->{}, params: {}->{}".format(flops, flopsnew, params, paramsnew))
    save_path=os.path.join(cfg.OUTPUT_DIR,"pruned_model.pth")
    torch.save(newmodel,save_path)
        
    # del model
    # del checkpointer
    # model=newmodel
    # logger = logging.getLogger('SSD.trainer')
    

    # device = torch.device(cfg.MODEL.DEVICE)
    # model.to(device)
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    # lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
    # optimizer = make_optimizer(cfg, model, lr)

    # milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    # scheduler = make_lr_scheduler(cfg, optimizer, milestones)
    # save_to_disk = dist_util.get_rank() == 0
    # checkpointer = CheckPointer(model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger)

    # max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus
    # train_loader = make_data_loader(cfg, is_train=True, distributed=args.distributed, max_iter=max_iter, start_iter=0)
    # arguments = {"iteration": 0}
    # args.sr=False
    # model=do_train(cfg, model,

    #          train_loader,
    #          optimizer,
    #          scheduler,
    #          checkpointer,
    #          device,
    #          arguments,
    #          args)






if __name__ == '__main__':
    main()