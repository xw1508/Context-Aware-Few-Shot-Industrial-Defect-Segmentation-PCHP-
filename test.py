import os
import random
import time
import cv2
import numpy as np
import logging
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data

from tensorboardX import SummaryWriter

try:
    from model.PCHP_test import net
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Private model package is not installed. Run `bash scripts/setup.sh` "
        "from the repository root, or install the wheel under `dist/` manually."
    ) from exc

from util import config, dataset, transform
from util.util import AverageMeter, intersectionAndUnionGPU

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

GPU_ID = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/SSD/fold0_resnet50_test.yaml', help='config file')
    parser.add_argument('--save_pred', action='store_true', help='save predicted masks')
    parser.add_argument('--save_vis', action='store_true', help='save predicted-mask overlays on query images')
    parser.add_argument('--save_prior', action='store_true', help='save prior probability maps if available in model.module._debug_prior_prob')
    args = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(args.config)

    # 把额外命令行参数并入 cfg，避免原有 yaml 失效
    cfg.save_pred = args.save_pred
    cfg.save_vis = args.save_vis
    cfg.save_prior = args.save_prior
    return cfg


def get_logger():
    logger_name = 'main-logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = '[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s'
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return True


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def tensor_mask_to_png(pred_mask_tensor):
    """
    pred_mask_tensor: [B,H,W] or [H,W], class index mask
    return: uint8 numpy mask in {0,255}
    """
    if pred_mask_tensor.dim() == 3:
        pred_mask_tensor = pred_mask_tensor[0]
    pred = pred_mask_tensor.detach().cpu().numpy().astype(np.uint8)
    pred[pred != 0] = 255
    pred[pred == 0] = 0
    return pred


def tensor_mask_to_numpy(pred_mask_tensor, batch_index=0):
    if pred_mask_tensor.dim() == 3:
        pred_mask_tensor = pred_mask_tensor[batch_index]
    pred = pred_mask_tensor.detach().cpu().numpy().astype(np.uint8)
    pred[pred != 0] = 1
    return pred


def query_paths_to_list(query_paths):
    if isinstance(query_paths, (list, tuple)):
        return list(query_paths)
    return [query_paths]


def save_query_overlay(query_img_path, pred_mask_tensor, save_path, batch_index=0, alpha=0.45):
    query_img = cv2.imread(query_img_path, cv2.IMREAD_COLOR)
    if query_img is None:
        logger.warning('Failed to read query image for overlay: {}'.format(query_img_path))
        return

    pred = tensor_mask_to_numpy(pred_mask_tensor, batch_index=batch_index)
    img_h, img_w = query_img.shape[:2]
    if pred.shape[:2] != (img_h, img_w):
        pred = cv2.resize(pred, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

    mask = pred > 0
    overlay = query_img.copy()
    color = np.zeros_like(query_img)
    color[:, :] = (0, 0, 255)
    blended = cv2.addWeighted(query_img, 1 - alpha, color, alpha, 0)
    overlay[mask] = blended[mask]
    cv2.imwrite(save_path, overlay)


def save_prior_map(prior_tensor, save_path):
    """
    prior_tensor: [B,1,H,W] or [1,H,W]
    保存为 0~255 灰度图
    """
    if prior_tensor is None:
        return
    if prior_tensor.dim() == 4:
        prior_tensor = prior_tensor[0, 0]
    elif prior_tensor.dim() == 3:
        prior_tensor = prior_tensor[0]

    prior = prior_tensor.detach().float().cpu().numpy()
    prior = np.clip(prior, 0.0, 1.0)
    prior = (prior * 255.0).round().astype(np.uint8)
    cv2.imwrite(save_path, prior)


def main():
    args = get_parser()
    assert args.classes > 1

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        cudnn.deterministic = True
        cudnn.benchmark = False
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)
    main_worker(args)


def main_worker(argss):
    global args
    args = argss

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model = net(
        layers=args.layers,
        classes=2,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        pretrained=True,
        shot=args.shot,
        vgg=args.vgg,
    )

    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info('=> creating model ...')
    logger.info('Classes: {}'.format(args.classes))

    model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight, map_location='cpu')
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            logger.info("=> loaded weight '{}'".format(args.weight))
            if len(missing) > 0 or len(unexpected) > 0:
                logger.warning('Missing keys: {}'.format(missing))
                logger.warning('Unexpected keys: {}'.format(unexpected))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    assert args.split in [0, 1, 2]

    if args.resized_val:
        val_transform = transform.Compose([
            transform.Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    else:
        val_transform = transform.Compose([
            transform.test_Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])

    val_data = dataset.SemData(
        split=args.split,
        shot=args.shot,
        data_root=args.data_root,
        data_list=args.val_list,
        transform=val_transform,
        mode='val')

    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler)

    validate(val_loader, model, criterion)


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    split_gap = 4
    class_intersection_meter = [0] * split_gap
    class_union_meter = [0] * split_gap

    if args.manual_seed is not None and args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    pred_dir = './result/fold{}'.format(args.split)
    overlay_dir = './result/fold{}_overlay'.format(args.split)
    prior_dir = './result/fold{}_prior'.format(args.split)
    if getattr(args, 'save_pred', False):
        ensure_dir(pred_dir)
    if getattr(args, 'save_vis', False):
        ensure_dir(overlay_dir)
    if getattr(args, 'save_prior', False):
        ensure_dir(prior_dir)

    model.eval()
    end = time.time()

    test_num = len(val_loader)
    assert test_num % args.batch_size_val == 0
    iter_num = 0

    with torch.no_grad():
        for e in range(1):
            for i, (input, target, s_input, s_mask, subcls, ori_label, query_paths) in enumerate(val_loader):
                if (iter_num - 1) * args.batch_size_val >= test_num:
                    break
                iter_num += 1

                data_time.update(time.time() - end)
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                s_input = s_input.cuda(non_blocking=True)
                s_mask = s_mask.cuda(non_blocking=True)
                ori_label = ori_label.cuda(non_blocking=True)

                start_time = time.time()

                # 关键适配点：maptnet1_29_4.py 在 eval() 下只返回 query_pred_mask logits，
                # 不会返回 (output, pred)
                output = model(s_x=s_input, s_y=s_mask, x=input, y=target)
                pred = output.max(1)[1]

                model_time.update(time.time() - start_time)

                if getattr(args, 'save_pred', False):
                    save_path = os.path.join(pred_dir, '{}.png'.format(i + 1))
                    cv2.imwrite(save_path, tensor_mask_to_png(pred))

                if getattr(args, 'save_prior', False):
                    prior_prob = getattr(model.module, '_debug_prior_prob', None)
                    if prior_prob is not None:
                        prior_save_path = os.path.join(prior_dir, '{}.png'.format(i + 1))
                        save_prior_map(prior_prob, prior_save_path)

                if args.ori_resize:
                    longerside = max(ori_label.size(1), ori_label.size(2))
                    backmask = torch.ones(ori_label.size(0), longerside, longerside).cuda() * 255
                    backmask[0, :ori_label.size(1), :ori_label.size(2)] = ori_label
                    target = backmask.clone().long()

                output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
                loss = criterion(output, target)
                loss = torch.mean(loss)

                output_label = output.max(1)[1]

                if getattr(args, 'save_vis', False):
                    query_path_list = query_paths_to_list(query_paths)
                    for batch_idx, query_path in enumerate(query_path_list):
                        sample_idx = i * args.batch_size_val + batch_idx + 1
                        base_name = os.path.splitext(os.path.basename(query_path))[0]
                        save_path = os.path.join(overlay_dir, '{:04d}_{}_overlay.png'.format(sample_idx, base_name))
                        save_query_overlay(query_path, output_label, save_path, batch_index=batch_idx)

                intersection, union, new_target = intersectionAndUnionGPU(
                    output_label, target, args.classes, args.ignore_label)
                intersection = intersection.cpu().numpy()
                union = union.cpu().numpy()
                target_np = target.cpu().numpy()
                new_target = new_target.cpu().numpy()

                intersection_meter.update(intersection)
                union_meter.update(union)
                target_meter.update(new_target)

                subcls = subcls[0].cpu().numpy()[0]
                class_intersection_meter[(subcls - 1) % split_gap] += intersection[1]
                class_union_meter[(subcls - 1) % split_gap] += union[1]

                accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
                loss_meter.update(loss.item(), input.size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                log_interval = max(1, int(test_num / 100))
                if ((i + 1) % log_interval == 0) and main_process():
                    logger.info(
                        'Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(
                            iter_num * args.batch_size_val,
                            test_num,
                            data_time=data_time,
                            batch_time=batch_time,
                            loss_meter=loss_meter,
                            accuracy=accuracy)
                    )

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i] / (class_union_meter[i] + 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou * 1.0 / len(class_intersection_meter)

    logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
    for i in range(split_gap):
        logger.info('Class_{} Result: iou {:.4f}.'.format(i + 1, class_iou_class[i]))

    if main_process():
        logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        print('avg inference time: {:.4f}, count: {}'.format(model_time.avg, test_num))
        logger.info('<<<<<<<<<<<<<<<<< End fold{},shot{} Evaluation <<<<<<<<<<<<<<<<<'.format(args.split, args.shot))

    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou


if __name__ == '__main__':
    main()
