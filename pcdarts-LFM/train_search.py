import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network, initialize_alphas
from architect import Architect, calculate_weightage
from image_encoder import *
from genotypes import PRIMITIVES


parser = argparse.ArgumentParser("cifar")
parser.add_argument(
    "--data", type=str, default="../data", help="location of the data corpus"
)
parser.add_argument(
    "--workers", type=int, default=2, help="number of workers to load dataset"
)
parser.add_argument(
    "--set", type=str, default="cifar10", help="location of the data corpus"
)
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument(
    "--learning_rate", type=float, default=0.1, help="init learning rate"
)
parser.add_argument(
    "--learning_rate_min", type=float, default=0.0, help="min learning rate"
)
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--epochs", type=int, default=45, help="num of training epochs")
parser.add_argument(
    "--init_channels", type=int, default=16, help="num of init channels"
)
parser.add_argument("--layers", type=int, default=8, help="total number of layers")
parser.add_argument(
    "--model_path", type=str, default="saved_models", help="path to save the model"
)
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
parser.add_argument(
    "--drop_path_prob", type=float, default=0.3, help="drop path probability"
)
parser.add_argument("--save", type=str, default="EXP", help="experiment name")
parser.add_argument("--seed", type=int, default=2, help="random seed")
parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
parser.add_argument(
    "--train_portion", type=float, default=0.5, help="portion of training data"
)
parser.add_argument(
    "--unrolled",
    action="store_true",
    default=False,
    help="use one-step unrolled validation loss",
)
parser.add_argument(
    "--arch_learning_rate",
    type=float,
    default=6e-4,
    help="learning rate for arch encoding",
)
parser.add_argument(
    "--arch_weight_decay",
    type=float,
    default=1e-3,
    help="weight decay for arch encoding",
)

parser.add_argument("--learning_rate_v", type=float, default=6e-4)
parser.add_argument("--weight_decay_v", type=float, default=1e-3)
parser.add_argument("--learning_rate_r", type=float, default=6e-4)
parser.add_argument("--weight_decay_r", type=float, default=1e-3)
parser.add_argument("--img_encoder_arch", type=str, default="18")
parser.add_argument("--debug", default=False, action="store_true")
parser.add_argument("--pretrain_steps", type=int, default=5)
parser.add_argument("--warmup_steps", type=int, default=15)
args = parser.parse_args()

args.save = "search-{}-{}".format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*.py"))


log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10
if args.set == "cifar100":
    CIFAR_CLASSES = 100


def main():
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("gpu device = %d" % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    (
        arch,
        alphas_normal,
        alphas_reduce,
        betas_normal,
        betas_reduce,
    ) = initialize_alphas()

    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()
    # model for pretraining
    model_pretrain = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model_pretrain = model_pretrain.cuda()

    model._arch_parameters = arch
    model.alphas_normal = alphas_normal
    model.alphas_reduce = alphas_reduce
    model.betas_normal = betas_normal
    model.betas_reduce = betas_reduce

    model_pretrain._arch_parameters = arch
    model_pretrain.alphas_normal = alphas_normal
    model_pretrain.alphas_reduce = alphas_reduce
    model_pretrain.betas_normal = betas_normal
    model_pretrain.betas_reduce = betas_reduce

    logging.info("param size of model = %fMB", utils.count_parameters_in_MB(model))

    if args.img_encoder_arch == "18":
        img_encoder_v = resnet18().cuda()
    elif args.img_encoder_arch == "34":
        img_encoder_v = resnet34().cuda()
    elif args.img_encoder_arch == "50":
        img_encoder_v = resnet50().cuda()
    elif args.img_encoder_arch == "101":
        img_encoder_v = resnet101().cuda()

    coeff_r = nn.Linear(args.batch_size, 1).cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    optimizer_pretrain = torch.optim.SGD(
        model_pretrain.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    optimizer_v = torch.optim.SGD(
        img_encoder_v.parameters(),
        args.learning_rate_v,
        momentum=args.momentum,
        weight_decay=args.weight_decay_v,
    )
    optimizer_r = torch.optim.SGD(
        coeff_r.parameters(),
        args.learning_rate_r,
        momentum=args.momentum,
        weight_decay=args.weight_decay_r,
    )

    if args.set == "cifar100":
        train_transform, valid_transform = utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(
            root=args.data, train=True, download=True, transform=train_transform
        )
    else:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform
        )

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
        drop_last=True,
        num_workers=args.workers,
    )
    valid_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True,
        drop_last=True,
        num_workers=args.workers,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min
    )
    scheduler_pretrain = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_pretrain,
        float(args.epochs + args.pretrain_steps),
        eta_min=args.learning_rate_min,
    )
    scheduler_v = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_v, float(args.epochs), eta_min=args.learning_rate_min
    )
    scheduler_r = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_r, float(args.epochs), eta_min=args.learning_rate_min
    )

    architect = Architect(model, img_encoder_v, coeff_r, optimizer_v, optimizer_r, args)

    for epoch in range(args.epochs + args.pretrain_steps):
        lr = scheduler.get_lr()[0]
        lr_pretrain = scheduler_pretrain.get_lr()[0]
        lr_v = scheduler_v.get_lr()[0]
        lr_r = scheduler_r.get_lr()[0]
        logging.info(
            "epoch %d lr %e lr_pretrain %e lr_v %e lr_r %e",
            epoch,
            lr,
            lr_pretrain,
            lr_v,
            lr_r,
        )

        if epoch >= args.pretrain_steps:
            genotype = model.genotype()
            logging.info("genotype = %s", genotype)
            print(F.softmax(model.alphas_normal, dim=-1))
            print(F.softmax(model.alphas_reduce, dim=-1))

        # training
        train_acc, train_obj = train(
            args,
            epoch,
            train_queue,
            valid_queue,
            model,
            model_pretrain,
            img_encoder_v,
            coeff_r,
            architect,
            criterion,
            optimizer,
            optimizer_pretrain,
            optimizer_v,
            optimizer_r,
            lr,
            lr_pretrain,
            lr_v,
            lr_r,
        )

        if epoch >= args.pretrain_steps:
            logging.info("train_acc %f", train_acc)
        else:
            logging.info("pretrain_acc %f", train_acc)

        if epoch >= args.pretrain_steps:
            scheduler_pretrain.step()
            scheduler.step()
            scheduler_v.step()
            scheduler_r.step()
        else:
            scheduler_pretrain.step()

        # validation
        if epoch >= args.pretrain_steps:
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info("valid_acc %f", valid_acc)

            utils.save(model, os.path.join(args.save, "weights.pt"))


def train(
    args,
    epoch,
    train_queue,
    valid_queue,
    model,
    model_pretrain,
    img_encoder_v,
    coeff_r,
    architect,
    criterion,
    optimizer,
    optimizer_pretrain,
    optimizer_v,
    optimizer_r,
    lr,
    lr_pretrain,
    lr_v,
    lr_r,
):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    if epoch >= args.pretrain_steps:
        model.train()
    model_pretrain.train()

    valid_queue_iter = iter(valid_queue)
    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        try:
            input_search, target_search = next(valid_queue_iter)
            valid_queue_iter = iter(valid_queue)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)

        input_search = input_search.cuda(non_blocking=True)
        target_search = target_search.cuda(non_blocking=True)

        if epoch >= args.warmup_steps + args.pretrain_steps:
            architect.step(
                input,
                target,
                input_search,
                target_search,
                model_pretrain,
                lr,
                lr_v,
                lr_r,
                optimizer,
                unrolled=args.unrolled,
            )

        if epoch >= args.pretrain_steps:
            assert (
                model_pretrain._arch_parameters[0] - model._arch_parameters[0]
            ).sum() == 0
            assert (
                model_pretrain._arch_parameters[1] - model._arch_parameters[1]
            ).sum() == 0

            # --------------------------------------------------
            # train the model for pretrain
            # --------------------------------------------------
            optimizer_pretrain.zero_grad()
            logits = model_pretrain(input)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model_pretrain.parameters(), args.grad_clip)
            optimizer_pretrain.step()

            # --------------------------------------------------
            # train the model for search
            # --------------------------------------------------
            optimizer.zero_grad()
            logits = model(input)
            loss = F.cross_entropy(logits, target, reduction="none")
            # calculate the weightage of each training example
            a = calculate_weightage(
                input,
                target,
                input_search,
                target_search,
                model_pretrain,
                img_encoder_v,
                coeff_r,
            )
            loss = (loss * a).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # --------------------------------------------------
            # evaluate trained model for search
            # --------------------------------------------------
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info("train %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)
        else:
            assert (
                model_pretrain._arch_parameters[0] - model._arch_parameters[0]
            ).sum() == 0
            assert (
                model_pretrain._arch_parameters[1] - model._arch_parameters[1]
            ).sum() == 0

            # --------------------------------------------------
            # train the model for pretrain
            # --------------------------------------------------
            optimizer_pretrain.zero_grad()
            logits = model_pretrain(input)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model_pretrain.parameters(), args.grad_clip)
            optimizer_pretrain.step()

            # --------------------------------------------------
            # evaluate pretrained model
            # --------------------------------------------------
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info(
                    "pretrain %03d %e %f %f", step, objs.avg, top1.avg, top5.avg
                )

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info("valid %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == "__main__":
    main()
