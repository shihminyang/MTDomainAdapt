import csv
import json
import numpy as np
import os
from os.path import join as PJ
import torch
from torch.utils.data import DataLoader

from datasets.VisDA_Dataset import ClassificationDataset, CityScapesDataset
from datasets.eval.eval_classification import classification_evaluation
from datasets.eval.eval_segmentation import fast_hist, per_class_iu

from trainer import Segment
from utils import config, Timer


def test_classification(trainer, dataloader, output_directory, domain, mode, trans=False):
    trainer.eval()
    output_directory = PJ(output_directory, f"classification", f"{mode}_{domain}")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    gt_file = PJ(output_directory, 'gt_result.txt')
    pred_file = PJ(output_directory, 'pred_result.txt')
    eval_file = PJ(output_directory, 'eval_result.csv')

    display_iter = len(dataloader) // 2
    # Record predicts and labels
    print("===== Start Classification =====")
    f_gt, f_pred = open(gt_file, 'w'), open(pred_file, 'w')
    for it, (images, labels, _) in enumerate(dataloader):
        if (it + 1) % display_iter == 0: print(f"[{it+1:04d}/{len(dataloader)}]")

        images = images.cuda().detach()
        labels = labels.view(-1).tolist()
        # Classify
        predicts = trainer.classify(images, domain)
        predicts = torch.argmax(predicts, 1).data.tolist()
        # Record predicts and labels
        for pred, label in zip(predicts, labels):
            f_pred.write(f"{str(pred)}\n")
            f_gt.write(f"{str(label)}\n")
    f_gt.close(), f_pred.close()

    # Evaluate (Using ground truth and predict result files)
    print("Start Evaluation")
    evaluate = classification_evaluation(gt_file, pred_file)
    mean_acc = evaluate.mean_predictions_accuracy
    categories, cls_acc = evaluate.names, evaluate.predictions_accuracy.tolist()

    # Show evaluate result
    print(f"mean accuracy: {mean_acc:.3f}%")
    for i, (category, acc) in enumerate(zip(categories, cls_acc)):
        print(f"{category:10}: {acc:.3f}%", end='    ')
        if i % 3 == 2: print()

    # Record evaluate result
    with open(eval_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(categories + ['mAP'])
        writer.writerow(cls_acc + [mean_acc])
    return mean_acc, cls_acc


def test_segmentation(trainer, dataloader, output_directory, domain, mode, epoch, trans=False):
    trainer.eval()
    # Create directory and file to record testing results
    output_directory = PJ(output_directory, f"segmentation", f"{mode}_{domain}")
    segment_directory = PJ(output_directory, 'segmentation_image')
    if not os.path.exists(segment_directory):
        os.makedirs(segment_directory)
    eval_file = PJ(output_directory, 'eval_result.csv')

    # Load category information
    info_file = "./datasets/VisDA_17/semantic_segmentation/CityScapes/info.json"
    with open(info_file, 'r') as fp:
        info = json.load(fp)
    num_class = np.int(info['classes']) + 1
    categories = np.array(info['label'], dtype=np.str).tolist()

    # For visualize result
    visual_images, visual_labels, visual_results = [], [], []

    # Record number of correct pixel in each class
    hist = np.zeros((num_class, num_class))
    display_iter = len(dataloader) // 5
    print("===== Start Semantic segmentation =====")
    for it, (images, labels, _) in enumerate(dataloader):
        # Segmentation
        labels = labels.cpu()
        images = images.cuda().detach()
        predicts = trainer.segment(images, domain).cpu()
        # Calculate correct pixel (diag are correct for each class)
        hist += fast_hist(labels.numpy().flatten(), predicts.numpy().flatten(), num_class)

        if (it + 1) % display_iter == 0:
            print(f"[{it+1:04d}/{len(dataloader)}]")
            visual_images.append(images.cpu())
            visual_labels.append(labels.cpu())
            visual_results.append(predicts.cpu())

    # Visualization
    visual_images = torch.cat(visual_images, dim=0)
    visual_labels = torch.cat(visual_labels, dim=0)
    visual_results = torch.cat(visual_results, dim=0)
    trainer.record_segment_images(visual_labels, visual_images, segment_directory,
                                  f"{epoch:03d}.jpg", visual_results, domain, _scale=0.25)

    print("Start Evaluation")
    # Drop out background
    hist = hist[:-1, :-1]

    IOUs = (per_class_iu(hist) * 100).tolist()
    mIOU = np.nanmean(IOUs)
    print(f"mean IOU: {mIOU:.3f}%")
    for i, (category, iou) in enumerate(zip(categories, IOUs)):
        print(f"{category:10}: {iou:.3f}%", end='    ')
        if i % 3 == 2: print()
    print()
    # Record evaluate result
    with open(eval_file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(categories + ['mIOU'])
        writer.writerow(IOUs + [mIOU])
    return mIOU, IOUs


if __name__ == '__main__':
    mode = "test"
    domain = "target"

    exp_name = "exp1_seg_source_only"
    checkpoint_dir = f"./results/{exp_name}/checkpoints"
    config_path = f"./configs/{exp_name}.yaml"
    config = config(config_path)
    output_directory = PJ('./results', config['exp_name'])
    num_workers = config['num_workers']
    trainer = Segment(config)
    # Temp resume
    trainer.cuda()
    checkpoint_directory = PJ(output_directory, "checkpoints")
    trainer.resume(config, checkpoint_directory)

    transform = None
    dataset = CityScapesDataset(config['data_root_seg'], config['data_list']['test'], transform=transform, mode='test')
    testloader = DataLoader(dataset=dataset, batch_size=config['test_size'], shuffle=False, num_workers=num_workers)
    test_segmentation(trainer, testloader, output_directory, domain, mode, epoch=-1)
