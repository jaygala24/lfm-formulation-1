import os
import sys
import time
import glob
import warnings
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search_imagenet import Network, initialize_alphas
from architect import Architect, calculate_weightage
from image_encoder import *
from genotypes import PRIMITIVES

parser = argparse.ArgumentParser("imagenet")
parser.add_argument(
    "--workers", type=int, default=4, help="number of workers to load dataset"
)
parser.add_argument(
    "--data",
    type=str,
    default="../data/imagenet_sampled",
    help="location of the data corpus",
)
parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
parser.add_argument(
    "--learning_rate", type=float, default=0.1, help="init learning rate"
)
parser.add_argument(
    "--learning_rate_min", type=float, default=0.0, help="min learning rate"
)
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
parser.add_argument("--report_freq", type=float, default=100, help="report frequency")
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
    "--unrolled",
    action="store_true",
    default=False,
    help="use one-step unrolled validation loss",
)
parser.add_argument(
    "--arch_learning_rate",
    type=float,
    default=6e-3,
    help="learning rate for arch encoding",
)
parser.add_argument(
    "--arch_weight_decay",
    type=float,
    default=1e-3,
    help="weight decay for arch encoding",
)
parser.add_argument(
    "--begin", type=int, default=35, help="warm up steps to update architecture"
)
parser.add_argument("--learning_rate_v", type=float, default=6e-3)
parser.add_argument("--weight_decay_v", type=float, default=1e-3)
parser.add_argument("--learning_rate_r", type=float, default=6e-3)
parser.add_argument("--weight_decay_r", type=float, default=1e-3)
parser.add_argument("--img_encoder_arch", type=str, default="18")
parser.add_argument("--debug", default=False, action="store_true")
parser.add_argument("--pretrain_steps", type=int, default=5)

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

warnings.filterwarnings("ignore")

data_dir = os.path.join(args.data)
# data preparation, we random sample 10% and 2.5% from training set(each class) as train and val, respectively.
# Note that the data sampling can not use torch.utils.data.sampler.SubsetRandomSampler as imagenet is too large
CLASSES = 1000


def main():
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    np.random.seed(args.seed)
    # torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    # logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)
    # dataset_dir = '/cache/'
    # pre.split_dataset(dataset_dir)
    # sys.exit(1)
    # dataset prepare
    traindir = os.path.join(data_dir, "train")
    valdir = os.path.join(data_dir, "val")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # dataset split
    train_data1 = dset.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    train_data2 = dset.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    # valid_data = dset.ImageFolder(
    #     valdir,
    #     transforms.Compose(
    #         [
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             transforms.ToTensor(),
    #             normalize,
    #         ]
    #     ),
    # )
    num_train = len(train_data1)
    num_val = len(train_data2)
    print("# images to train network: %d" % num_train)
    print("# images to validate network: %d" % num_val)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    (
        arch,
        alphas_normal,
        alphas_reduce,
        betas_normal,
        betas_reduce,
    ) = initialize_alphas()

    model = Network(args.init_channels, CLASSES, args.layers, criterion)
    model_pretrain = Network(args.init_channels, CLASSES, args.layers, criterion)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

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

    model = torch.nn.DataParallel(model)
    model_pretrain = torch.nn.DataParallel(model_pretrain)
    model = model.cuda()
    model_pretrain = model_pretrain.cuda()

    if args.img_encoder_arch == "18":
        img_encoder_v = resnet18()
    elif args.img_encoder_arch == "34":
        img_encoder_v = resnet34()
    elif args.img_encoder_arch == "50":
        img_encoder_v = resnet50()
    elif args.img_encoder_arch == "101":
        img_encoder_v = resnet101()

    img_encoder_v = torch.nn.DataParallel(img_encoder_v)
    img_encoder_v = img_encoder_v.cuda()

    coeff_r = nn.Linear(args.batch_size, 1)
    coeff_r = torch.nn.DataParallel(coeff_r)
    coeff_r = coeff_r.cuda()

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
    optimizer_a = torch.optim.Adam(
        model.module.arch_parameters(),
        lr=args.arch_learning_rate,
        betas=(0.5, 0.999),
        weight_decay=args.arch_weight_decay,
    )

    # test_queue = torch.utils.data.DataLoader(
    #     valid_data,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     pin_memory=True,
    #     num_workers=args.workers,
    # )

    train_queue = torch.utils.data.DataLoader(
        train_data1,
        batch_size=args.batch_size,
        shuffle=True,
        # pin_memory=True,
        drop_last=True,
        num_workers=args.workers,
        prefetch_factor=2 * args.workers,
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data2,
        batch_size=args.batch_size,
        shuffle=True,
        # pin_memory=True,
        drop_last=True,
        num_workers=args.workers,
        prefetch_factor=2 * args.workers,
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

    # architect = Architect(model, args)
    lr = args.learning_rate
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
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer_pretrain.param_groups:
                param_group["lr"] = lr * (epoch + 1) / 5.0
            logging.info(
                "Warming-up Pretrain Epoch: %d, LR: %e", epoch, lr * (epoch + 1) / 5.0
            )
            # print(optimizer_pretrain)
        if (
            epoch >= args.pretrain_steps
            and epoch < 5 + args.pretrain_steps
            and args.batch_size > 256
        ):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr * (epoch - args.pretrain_steps + 1) / 5.0
            logging.info(
                "Warming-up Epoch: %d, LR: %e",
                epoch,
                lr * (epoch - args.pretrain_steps + 1) / 5.0,
            )
            # print(optimizer)

        if epoch >= args.pretrain_steps:
            genotype = model.module.genotype()
            logging.info("genotype = %s", genotype)
            # arch_param = model.module.arch_parameters()
            # logging.info(F.softmax(arch_param[0], dim=-1))
            # logging.info(F.softmax(arch_param[1], dim=-1))

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
            criterion,
            optimizer,
            optimizer_pretrain,
            optimizer_v,
            optimizer_r,
            optimizer_a,
            lr,
            lr_pretrain,
            lr_v,
            lr_r,
        )

        if epoch >= args.pretrain_steps:
            scheduler_pretrain.step()
            scheduler.step()
        else:
            scheduler_pretrain.step()

        if epoch >= args.pretrain_steps:
            logging.info("train_acc %f", train_acc)
        else:
            logging.info("pre_train_acc %f", train_acc)

        # validation
        if epoch >= 42 + args.pretrain_steps:
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            # test_acc, test_obj = infer(test_queue, model, criterion)
            logging.info("valid_acc %f", valid_acc)
            # logging.info('Test_acc %f', test_acc)

        # utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(
    args,
    epoch,
    train_queue,
    valid_queue,
    model,
    model_pretrain,
    img_encoder_v,
    coeff_r,
    criterion,
    optimizer,
    optimizer_pretrain,
    optimizer_v,
    optimizer_r,
    optimizer_a,
    lr,
    lr_pretrain,
    lr_v,
    lr_r,
):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    valid_queue_iter = iter(valid_queue)
    for step, (input, target) in enumerate(train_queue):
        if epoch >= args.pretrain_steps:
            model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)

        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)

        if epoch >= args.begin + args.pretrain_steps:
            # update the architecture A
            optimizer_a.zero_grad()
            logits = model(input_search)
            loss_a = criterion(logits, target_search)
            loss_a.sum().backward()
            nn.utils.clip_grad_norm_(model.module.arch_parameters(), args.grad_clip)
            optimizer_a.step()

            # update the image encoder V and coeff r
            optimizer_v.zero_grad()
            optimizer_r.zero_grad()

            with torch.no_grad():
                logits_train = model(input)
            train_loss = F.cross_entropy(logits_train, target, reduction="none")
            a = calculate_weightage(
                input,
                target,
                input_search,
                target_search,
                model_pretrain,
                img_encoder_v,
                coeff_r,
            )
            train_loss = (train_loss * a).sum()
            train_loss.backward()

            optimizer_v.step()
            optimizer_r.step()

        if epoch >= args.pretrain_steps:
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
            input = input.cuda()
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
