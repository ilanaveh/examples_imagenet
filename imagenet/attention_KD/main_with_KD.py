"""
10/12/25
Add Attention-based distillation to training
"""

import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from PIL import ImageFilter  # IN: for GaussianBlur
from tensorboardX import SummaryWriter
import random  # for GaussianBlurRand
from pathlib import Path

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='/home/projects/bagon/shared/imagenet',
                    help='path to imagenet dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet101',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18. IN: changed to resnet101)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4. IN: changed to 8)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10. IN: changed to 100)')
parser.add_argument('--resume',
                    default='/home/projects/bagon/ilanaveh/code/examples_imagenet/imagenet/attention_KD/out',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none. IN: changed to out directory)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--no-accel', action='store_true',
                    help='disables accelerator')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
# IN: added arguments for training with blurry inputs (and blur_max for variable-blur training)
parser.add_argument('--blur', default=0, type=int, help="blur level for training")
parser.add_argument('--blur_max', default=None, type=int, help='For Variable-Blur training: max sigma')
parser.add_argument('--suf', default='', type=str, help='Suffix for model name')
parser.add_argument('--tb_subdir', default='', type=str,
                    help='If tensorboard file should be saved in subdir within X_epochs')
# IN: Add argument for changing kernel size of first convolutional layer, for reliable RF analysis (ref: Pawan 2018)
parser.add_argument('--conv1_ker_size', default=None, type=int, help='If not None, changes size of conv1 kernel.')
parser.add_argument('--conv1_stride', default=None, type=int, help='If not None, changes stride of conv1 kernel.')
# 10/12/25: Add Arguments for distillation:
parser.add_argument('--tchr_path', default=None, type=str, help='Path to teacher model (typically high-res)')
parser.add_argument('--alpha', default=1.0, type=float, help='Alpha parameter for KD loss')
parser.add_argument('--kd_layers', default=[1, 2, 3, 4], type=int, nargs='+',
                    help='Resnet layers for attention-transfer; choose from [1, 2, 3, 4]')

best_acc1 = 0


# =========================
# Attention Distillation Utils
# =========================

class FeatureHook:
    def __init__(self):
        self.features = None

    def __call__(self, module, input, output):
        self.features = output

    def clear(self):
        self.features = None


def attention_map(feat):
    # feat: B x C x H x W
    return feat.pow(2).mean(dim=1, keepdim=True)


def normalize_attention(att):
    att = att.view(att.size(0), -1)
    return torch.nn.functional.normalize(att, p=2, dim=1)


def attention_distill_loss(student_feats, teacher_feats):
    loss = 0.
    for fs, ft in zip(student_feats, teacher_feats):
        As = normalize_attention(attention_map(fs))
        At = normalize_attention(attention_map(ft))
        loss += torch.nn.functional.mse_loss(As, At)
    return loss


def main():
    is_db = torch.cuda.device_count() == 1
    args = parser.parse_args()
    args.model_name = f'train_resnet_blur{args.blur}'
    args.model_name = args.model_name + '-{}'.format(args.blur_max) if args.blur_max else args.model_name
    args.model_name = args.model_name + '_ker{}'.format(args.conv1_ker_size) if args.conv1_ker_size else args.model_name
    args.model_name = args.model_name + '_stride{}'.format(args.conv1_stride) if args.conv1_stride else args.model_name
    args.model_name = args.model_name + '_KD' if args.tchr_path else args.model_name
    args.model_name = args.model_name + '_alpha{}'.format(args.alpha) if args.tchr_path else args.model_name
    args.model_name = args.model_name + '_lyrs{}'.format(''.join(str(l) for l in args.kd_layers)) if args.tchr_path \
        else args.model_name
    args.model_name = args.model_name + '_{}'.format(args.suf) if args.suf else args.model_name
    args.model_name = args.model_name + '_db' if is_db else args.model_name

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # --- torchrun / DDP detection ---
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

        dist.init_process_group(
            backend=args.dist_backend,
            init_method="env://"
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    args.distributed = distributed

    is_main_process = (not args.distributed) or dist.get_rank() == 0

    print_if_main(f"~~~{args.model_name}~~~", is_main_process)

    main_worker(device, args, is_main_process)


def main_worker(device, args, is_main_process):
    global best_acc1

    tb_dir = "/home/projects/bagon/ilanaveh/code/examples_imagenet/imagenet/attention_KD/board/{}_epochs".format(
        args.epochs)
    tb_dir = os.path.join(tb_dir, args.tb_subdir) if args.tb_subdir else tb_dir

    if is_main_process:
        writer_tb = SummaryWriter(log_dir=os.path.join(tb_dir, args.model_name))
        print("=> Tensorboard saved at: {}".format(os.path.join(tb_dir, args.model_name)))
    else:
        writer_tb = None

    # create model
    if args.pretrained:
        print_if_main("=> Using pre-trained model '{}'".format(args.arch), is_main_process)
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print_if_main("=> Creating model '{}'".format(args.arch), is_main_process)
        model = models.__dict__[args.arch]()

        if args.conv1_ker_size is not None:
            ori_conv1_ker_size = model.conv1.weight.shape[-1]
            print_if_main(f"=> Changing conv1 ker size from: {ori_conv1_ker_size} to: {args.conv1_ker_size}",
                          is_main_process)
            if args.conv1_stride is not None:
                stride = args.conv1_stride
                print_if_main(f"=> Changing conv1 stride from 2 to {args.conv1_stride}", is_main_process)
            else:
                stride = 2
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(args.conv1_ker_size, args.conv1_ker_size),
                                    stride=(stride, stride), padding=int((args.conv1_ker_size - 1) / 2), bias=False)

    # =========================
    # Load Teacher Model
    # =========================
    use_kd = False
    if args.tchr_path:
        tchr_file = os.path.join(args.tchr_path, 'model_best.pth.tar')
        if os.path.isfile(tchr_file):
            use_kd = True
            print_if_main(f"=> Creating teacher model: {tchr_file}", is_main_process)
            tchr_model = models.__dict__[args.arch]()
            if args.conv1_ker_size is not None:
                ori_conv1_ker_size = tchr_model.conv1.weight.shape[-1]
                print_if_main(f"=> Teacher model: Changing conv1 ker size from: {ori_conv1_ker_size} to: {args.conv1_ker_size}",
                      is_main_process)
                if args.conv1_stride is not None:
                    stride = args.conv1_stride
                    print_if_main(f"=> Teacher model: Changing conv1 stride from 2 to {args.conv1_stride}", is_main_process)
                else:
                    stride = 2
                tchr_model.conv1 = nn.Conv2d(3, 64, kernel_size=(args.conv1_ker_size, args.conv1_ker_size),
                                             stride=(stride, stride), padding=int((args.conv1_ker_size - 1) / 2),
                                             bias=False)

            # Change to dataparallel, to enable loading from checkpoint:
            tchr_model = torch.nn.DataParallel(tchr_model)

            tchr_model.to(device)

            if args.gpu is None:
                tchr_checkpoint = torch.load(tchr_file)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f'{device.type}:{args.gpu}'
                tchr_checkpoint = torch.load(tchr_file, map_location=loc)
            tchr_best_epoch = tchr_checkpoint['epoch']
            tchr_best_acc1 = tchr_checkpoint['best_acc1']
            tchr_model.load_state_dict(tchr_checkpoint['state_dict'])
            tchr_model = tchr_model.module
            print_if_main(f"=> Teacher model: loaded checkpoint (epoch {tchr_best_epoch}, acc1: {tchr_best_acc1})",
                  is_main_process)

            tchr_model.eval()

            # Freeze teacher
            for p in tchr_model.parameters():
                p.requires_grad = False

        else:
            print_if_main("=> Teacher model: no checkpoint found at '{}' => Not using distillation.".format(tchr_file),
                  is_main_process)
            tchr_model = None
    else:
        print_if_main("=> No Teacher path given => Not using distillation.", is_main_process)

    if args.distributed:
        print_if_main("=> Using DistributedDataParallel (torchrun)", is_main_process)

        # one process = one GPU
        model = model.to(device)

        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False
        )

    else:
        # single-GPU fallback (NO DataParallel)
        print("=> Using single GPU (no DataParallel)")
        model = model.to(device)

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # optionally resume from a checkpoint
    if args.resume:
        output_dir = Path(args.resume) / args.model_name
        output_dir.mkdir(parents=False, exist_ok=True)  # create if doesn't exist, alert if parent doesn't exist.
        resume_checkpoint_file = os.path.join(output_dir, 'checkpoint.pth.tar')
        if os.path.isfile(resume_checkpoint_file):
            print_if_main("=> Loading checkpoint '{}'".format(resume_checkpoint_file), is_main_process)
            if args.gpu is None:
                checkpoint = torch.load(resume_checkpoint_file)
            else:
                # Map model to be loaded to specified single gpu.
                loc = f'{device.type}:{args.gpu}'
                checkpoint = torch.load(resume_checkpoint_file, map_location=loc)
            args.start_epoch = checkpoint['epoch'] + 1
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print_if_main("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']), is_main_process)
        else:
            print_if_main("=> No checkpoint found at '{}'".format(resume_checkpoint_file), is_main_process)

    # Data loading code
    if args.dummy:
        print("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        # IN: add blur transform

        # original blur lists:
        post_blur_transforms = {
            'train': [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ],
            'val': [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ]
        }

        if args.blur or args.blur_max:
            transforms_fin = {}
            if args.blur_max:

                blur_transform = GaussianBlurRand(int(args.blur), int(args.blur_max))
                max_blur_transform = GaussianBlur(int(args.blur_max))  # for validation
                transforms_fin['val'] = transforms.Compose([max_blur_transform] + post_blur_transforms['val'])
            else:
                blur_transform = GaussianBlur(int(args.blur))
                transforms_fin['val'] = transforms.Compose([blur_transform] + post_blur_transforms['val'])

            transforms_fin['train'] = transforms.Compose([blur_transform] + post_blur_transforms['train'])

        else:
            transforms_fin = {X: transforms.Compose(post_blur_transforms[X]) for X in ['train', 'val']}

        print_if_main('=> Creating datasets.', is_main_process)
        train_dataset = datasets.ImageFolder(traindir, transforms_fin['train'])

        val_dataset = datasets.ImageFolder(valdir, transforms_fin['val'])
        print_if_main('=> Datasets created.', is_main_process)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, sampler=val_sampler)

    # =========================
    # Add hooks for distillation
    # =========================
    tchr_hooks = []
    stdnt_hooks = []

    def get_last_conv(model, layer_idx):
        block = getattr(model, f'layer{layer_idx}')[-1]
        return block.conv3 if hasattr(block, 'conv3') else block.conv2

    if use_kd:
        for i in args.kd_layers:
            ht = FeatureHook()
            hs = FeatureHook()

            get_last_conv(tchr_model, i).register_forward_hook(ht)
            if args.distributed:
                get_last_conv(model.module, i).register_forward_hook(hs)
            else:
                get_last_conv(model, i).register_forward_hook(hs)

            tchr_hooks.append(ht)
            stdnt_hooks.append(hs)

    if args.evaluate:
        print_if_main('=> Running validation.', is_main_process)
        validate(val_loader, model, criterion, args)
        return

    print_if_main(f'=> Starting train loop from epoch {args.start_epoch}.', is_main_process)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_stats = train(train_loader, model, criterion, optimizer, epoch, device, args,
                            tchr_model, tchr_hooks, stdnt_hooks, is_main_process)

        if writer_tb is not None:
            print('Writing TB Train, epoch {}\n'.format(epoch))
            writer_tb.add_scalar('Loss/Train_Loss', train_stats['loss'], epoch)
            writer_tb.add_scalar('Top1/Train_Top1', train_stats['acc1'], epoch)

        # evaluate on validation set
        val_stats = validate(val_loader, model, criterion, args)

        if writer_tb is not None:
            print('Writing TB Val, epoch {}\n'.format(epoch))
            writer_tb.add_scalar('Loss/Val_Loss', val_stats['loss'], epoch)
            writer_tb.add_scalar('Top1/Val_Top1', val_stats['acc1'], epoch)

        scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = val_stats['acc1'] > best_acc1
        best_acc1 = max(val_stats['acc1'], best_acc1)

        if is_best:
            print_if_main(f"New best top-1 accuracy! (epoch {epoch}, acc1={best_acc1})", is_main_process)

        save_checkpoint_file_name = resume_checkpoint_file if args.resume else 'checkpoint.pth.tar'
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, is_best, filename=save_checkpoint_file_name, is_main=is_main_process)


def train(train_loader, model, criterion, optimizer, epoch, device, args, tchr_model, tchr_hooks, stdnt_hooks, is_main):
    print_if_main(f"=> Epoch {epoch}", is_main)
    use_accel = not args.no_accel and torch.accelerator.is_available()

    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    data_time = AverageMeter('Data', use_accel, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.NONE)
    top5 = AverageMeter('Acc@5', use_accel, ':6.2f', Summary.NONE)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # teacher forward (no grad)
        if tchr_model:
            with torch.no_grad():
                tchr_model(images)

        # student forward
        output = model(images)

        # Original loss
        cls_loss = criterion(output, target)

        # KD loss
        if tchr_model:
            att_loss = attention_distill_loss(
                [h.features for h in stdnt_hooks],
                [h.features for h in tchr_hooks]
            )

            # Combined weighted loss
            loss = cls_loss + args.alpha * att_loss

            # print_if_main(cls_loss.item(), att_loss.item(), is_main)
        else:
            loss = cls_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # clear hooks after each step
        if tchr_model:
            for h in tchr_hooks + stdnt_hooks:
                h.clear()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

    return {'loss': losses.avg, 'acc1': top1.avg}


def validate(val_loader, model, criterion, args):
    use_accel = not args.no_accel and torch.accelerator.is_available()

    def run_validate(loader, base_progress=0):

        if use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if use_accel:
                    if args.gpu is not None and device.type == 'cuda':
                        torch.accelerator.set_device_index(args.gpu)
                        images = images.cuda(args.gpu, non_blocking=True)
                        target = target.cuda(args.gpu, non_blocking=True)
                    else:
                        images = images.to(device)
                        target = target.to(device)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', use_accel, ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return {'loss': losses.avg, 'acc1': top1.avg}


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', is_main=True):
    print_if_main(f'Saving checkpoint (epoch {state["epoch"]}) at: {filename}', is_main)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('checkpoint', 'model_best'))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, use_accel, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.use_accel = use_accel
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if self.use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class GaussianBlur(object):
    """Apply Gaussian blur filter with the given sigma to the input PIL Image.
    Args:
        sigma (int): Desired Gaussian blur level sigma

    Taken from: W:/dannyh/work/code/PyTorch/vggface2_lookdir/datasets/custom_transforms.
   """

    def __init__(self, sigma):
        assert isinstance(sigma, int)
        self.sigma = sigma

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        img = img.filter(ImageFilter.GaussianBlur(radius=self.sigma))

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={0})'.format(self.sigma)


class GaussianBlurRand(object):
    """
    Apply Gaussian blur filter to the input PIL Image, with a rondom choice between self.sigma_min-self.sigma_max.
    if no sigma_max is given (or if sigma_min = sigma_max) -> same as regular GaussianBlur.
    Taken from: DeepLabv3FineTuning-disClasses/pretraining_resnet/pretrain_resnet_var_blurs.py.
    Args:
        sigma_min (int): Desired Gaussian blur level sigma / lower bound
        sigma_max (int; optional): Upper bound.
   """

    def __init__(self, sigma_min=0, sigma_max=None):
        assert isinstance(sigma_min, int)
        self.is_range = bool(sigma_max) & (sigma_min != sigma_max)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img, return_blur=False):
        """
        Args:
            img (PIL Image): Image to be scaled.
            return_blur (bool): Whether to return the chosen blur sigma.
        Returns:
            PIL Image: Rescaled image.
            if return_blur=True: also return the chosen blur sigma.
        """

        radius = random.randint(self.sigma_min, self.sigma_max) if self.is_range else self.sigma_min
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        if return_blur:
            return img, radius
        else:
            return img

    def __repr__(self):
        if self.is_range:
            return self.__class__.__name__ + '(sigma={}-{})'.format(self.sigma_min, self.sigma_max)
        else:
            return self.__class__.__name__ + '(sigma={})'.format(self.sigma_min)


def print_if_main(msg, is_main_process):
    if is_main_process:
        print(msg)


if __name__ == '__main__':
    main()
