from os.path import join as PJ
import shutil
import sys
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from datasets.VisDA_Dataset import ClassificationDataset, CityScapesDataset, GTA5Dataset
from trainer import Classify, Segment, Multi_task
from adapt_trainer import ClassifyAdapt, SegmentAdapt, MTAdapt
from test import test_classification, test_segmentation
from utils import config, write_loss, Timer, transform_setup, prepare_folder


def train_classification(dataloader_list):
    print("Train Classification\n")
    trainloader_s, valloader_s, trainloader_t, valloader_t = dataloader_list
    # Resume training process
    iterations = trainer.resume(config, checkpoint_directory) if resume_train else 0
    epoch = iterations // len(trainloader_s)
    while True:
        for it, data in enumerate(trainloader_s):
            trainer.update_learning_rate()

            images = data[0].cuda().detach()
            labels = data[1].cuda().detach()
            # Main training code
            display_time = True if (iterations + 1) % config['log_iter'] == 0 else False
            with Timer("Elapsed time in update:", display_time, images.shape[2:]):
                trainer.model_update(images, labels, config)

            # Record training log in SummaryWriter
            if display_time:
                print(f"Iteration: [{iterations+1:08d}/{max_iter:08d}]  (Epoch: {epoch+1})")
                write_loss(iterations, trainer, train_writer)
                train_writer.add_scalar("lr", trainer.optimizer.param_groups[0]['lr'], iterations + 1)

            # Save network weights for resume
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(config, checkpoint_directory, iterations + 1)

            ########################################
            # Validation (eval_iter)
            ########################################
            if (iterations + 1) % config['eval_iter'] == 0:
                print(f"\n{config['exp_name']}  (Epoch {epoch+1})")
                trainer.eval()
                if (iterations + 1) % (3 * config['eval_iter']) == 0:
                    mean_acc_s, _ = test_classification(trainer, valloader_s, output_directory, 'source', 'val')
                    train_writer.add_scalar("Evaluate/Source/mean_acc", mean_acc_s, iterations + 1)
                mean_acc_t, _ = test_classification(trainer, valloader_t, output_directory, 'target', 'val')
                train_writer.add_scalar("Evaluate/Target/mean_acc", mean_acc_t, iterations + 1)
                trainer.train()

            if (iterations + 1) % len(trainloader_s) == 0:
                epoch += 1
            iterations += 1
            if iterations >= max_iter or epoch >= config['max_epoch']:
                print(f"\n{config['exp_name']}")
                sys.exit('Finish training')


def train_segmentation(dataloader_list):
    print("Train Segmentation\n")
    trainloader_s, valloader_s, trainloader_t, valloader_t = dataloader_list
    # Resume training process
    iterations = trainer.resume(config, checkpoint_directory) if resume_train else 0
    epoch = iterations // len(trainloader_s)
    while True:
        for it, data in enumerate(trainloader_s):
            trainer.update_learning_rate()
            images = data[0].cuda().detach()
            labels = data[1].cuda().detach()

            # Main training code
            display_time = True if (iterations + 1) % config['log_iter'] == 0 else False
            with Timer("Elapsed time in update:", display_time, images.shape[2:]):
                trainer.model_update(images, labels, config)

            # Record training log in SummaryWriter
            if display_time:
                print(f"Iteration: [{iterations+1:08d}/{max_iter:08d}]  (Epoch: {epoch+1})")
                write_loss(iterations, trainer, train_writer)
                train_writer.add_scalar("lr", trainer.optimizer.param_groups[0]['lr'], iterations + 1)

            # Record current semantic segmentation images
            if (iterations + 1) % config['image_display_iter'] == 0:
                trainer.record_segment_images(labels, images, segment_directory, "current.jpg")

            # Record semantic segmentation images
            if (iterations + 1) % config['image_save_iter'] == 0:
                name = f"{epoch+1:03d}_{iterations+1:08d}.jpg"
                trainer.record_segment_images(labels, images, segment_directory, name)

            # Save network weights for resume
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(config, checkpoint_directory, iterations + 1)

            ########################################
            # Validation (Epoch)
            ########################################
            if (iterations + 1) % config['eval_iter'] == 0:
                print(f"\n{config['exp_name']}  (Epoch {epoch+1})")

                trainer.eval()
                # Target domain
                mIOU, IOUs = test_segmentation(trainer, valloader_t, output_directory, 'target', 'val', epoch)
                train_writer.add_scalar("Evaluate/Target/mIOU", mIOU, iterations + 1)

                # Not to always evaluate source (much more data)
                if (iterations + 1) % (3 * config['eval_iter']) == 0:
                    mIOU, IOUs = test_segmentation(trainer, valloader_s, output_directory, 'source', 'val', epoch)
                    train_writer.add_scalar("Evaluate/Source/mIOU", mIOU, iterations + 1)
                trainer.train()

            if (iterations + 1) % len(trainloader_s) == 0:
                epoch += 1

            iterations += 1
            if iterations >= max_iter or epoch >= config['max_epoch']:
                print(f"\n{config['exp_name']}")
                sys.exit('Finish training')


def train_adaptation(dataloader_list):
    print("Train Adaptation\n")
    trainloader_s, valloader_s, trainloader_t, valloader_t = dataloader_list

    # Resume training process
    iterations = trainer.resume(config, checkpoint_directory) if resume_train else 0
    epoch = iterations // len(trainloader_s)
    while True:
        for it, datas in enumerate(zip(trainloader_s, trainloader_t)):
            trainer.update_learning_rate()

            source_cls, target_cls = datas
            images_s = source_cls[0].cuda().detach()
            images_t = target_cls[0].cuda().detach()
            labels_s = source_cls[1].cuda().detach()

            # Main training code
            display_time = True if (iterations + 1) % config['log_iter'] == 0 else False
            with Timer("Elapsed time in update:", display_time, images_s.shape[2:]):
                trainer.dis_update(images_s, images_t, config)
                trainer.gen_update(images_s, images_t, labels_s, config)
                torch.cuda.synchronize()    # Wait for all operations done

            # Record training log in SummaryWriter
            if display_time:
                print(f"Iteration: [{iterations+1:08d}/{max_iter:08d}]  (Epoch: {epoch+1})")
                write_loss(iterations, trainer, train_writer)
                train_writer.add_scalar("lr", trainer.gen_opt.param_groups[0]['lr'], iterations + 1)

            # Record current translate images
            if (iterations + 1) % config['image_display_iter'] == 0:
                trainer.record_translate_images(images_s, images_t, translate_directory, "current")
                # trainer.record_segment_images(labels_s, images_s, 'source', segment_directory, "current")

            # Record translate images
            if (iterations + 1) % config['image_save_iter'] == 0:
                name = f"{epoch+1:03d}_{iterations+1:08d}"
                trainer.record_translate_images(images_s, images_t, translate_directory, name)
                # trainer.record_segment_images(labels_s, images_s, 'source', segment_directory, name)

            # Save network weights for resume
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(config, checkpoint_directory, iterations + 1)

            ########################################
            # Validation
            ########################################
            if (iterations + 1) % config['eval_iter'] == 0:
                print(f"\n{config['exp_name']}  (Epoch {epoch+1})")
                trainer.eval()
                # Target domain
                mIOU, IOUs = test_segmentation(trainer, valloader_t, output_directory, 'target', 'val', epoch)
                train_writer.add_scalar("Evaluate/Target/mIOU", mIOU, iterations + 1)
                trainer.train()

            if (iterations + 1) % len(trainloader_s) == 0:
                epoch += 1

            iterations += 1
            if iterations >= max_iter or epoch >= config['max_epoch']:
                print(f"\n{config['exp_name']}")
                sys.exit('Finish training')


def train_multi_task_adapt(dataloader_list):
    trainer.eval()
    torch.set_grad_enabled(False)

    print("Multi-Task")
    trainloader_cls_s, valloader_cls_s, trainloader_cls_t, valloader_cls_t = dataloader_list[:4]
    trainloader_seg_s, valloader_seg_s, trainloader_seg_t, valloader_seg_t = dataloader_list[4:]

    iter_cls_s, iter_cls_t = iter(trainloader_cls_s), iter(trainloader_cls_t)
    iter_seg_s, iter_seg_t = iter(trainloader_seg_s), iter(trainloader_seg_t)
    # Resume training process
    iterations = trainer.resume(config, checkpoint_directory) if resume_train else 0
    print(f"iterations start in {iterations}")
    epoch_cls = iterations // len(trainloader_cls_s)
    epoch_seg = iterations // len(trainloader_seg_s)
    while True:
        try:
            source_cls = iter_cls_s.next()
        except StopIteration:
            iter_cls_s = iter(trainloader_cls_s)
            source_cls = iter_cls_s.next()

        try:
            target_cls = iter_cls_t.next()
        except StopIteration:
            iter_cls_t = iter(trainloader_cls_t)
            target_cls = iter_cls_t.next()

        try:
            source_seg = iter_seg_s.next()
        except StopIteration:
            iter_seg_s = iter(trainloader_seg_s)
            source_seg = iter_seg_s.next()

        try:
            target_seg = iter_seg_t.next()
        except StopIteration:
            iter_seg_t = iter(trainloader_seg_t)
            target_seg = iter_seg_t.next()

        images_cls_s = source_cls[0].cuda().detach()
        images_cls_t = target_cls[0].cuda().detach()
        images_seg_s = source_seg[0].cuda().detach()
        images_seg_t = target_seg[0].cuda().detach()

        name = f"{epoch_seg+1:03d}_{iterations+1:08d}"
        trainer.record_translate_images(images_cls_s, images_cls_t, translate_directory, f"{name}_cls")
        trainer.record_translate_images(images_seg_s, images_seg_t, translate_directory, f"{name}_seg")
        # if config['task'] == 'Segmentation' or config['task'] == 'Multi_task':
        #     trainer.record_segment_images(labels_seg_s, images_seg_s, segment_directory, name, domain='source')

        if (iterations + 1) % len(trainloader_cls_s) == 0:
            epoch_cls += 1
        if (iterations + 1) % len(trainloader_seg_s) == 0:
            epoch_seg += 1

        iterations += 1
        if iterations >= max_iter or epoch_cls >= config['max_epoch']:
            print(f"\n{config['exp_name']}")
            sys.exit('Finish training')


########################################
# Environment and Experiment setting
########################################
# Here to change experiments
config_path = "./configs/exp3_mtl_adatp_step1.yaml"
config = config(config_path)

resume_train = config['resume']
max_iter = config['max_iter']
num_workers = config['num_workers']
output_directory = PJ('./results', config['exp_name'])
checkpoint_directory, translate_directory, segment_directory = prepare_folder(config, output_directory)
train_writer = SummaryWriter(PJ('./results/logs', config['exp_name']))

shutil.copy(config_path, PJ(output_directory, f"{config['exp_name']}.yaml"))
print(f"\n{config['exp_name']}")
print(f"Task: {config['task']}")
print(f"Trainer: {config['trainer']}", end=" | ")
print(f"Classifier: {config['classifier']} | Segment: {config['segmentor']}\n")

########################################
# Classification data loader
########################################
if config['classifier'] is not None:
    # Source domain dataset
    transform = transform_setup(config, "classification", mode="train")
    trainset_cls_s = ClassificationDataset(config['data_root_cls'], config['data_list_s']['train'], transform=transform, domain='source')
    transform = transform_setup(config, "classification", mode="val")
    valset_cls_s = ClassificationDataset(config['data_root_cls'], config['data_list_s']['val'], transform=transform, domain='source')
    # Dataloader
    trainloader_cls_s = DataLoader(dataset=trainset_cls_s, batch_size=config['cls_batch_size'], shuffle=True, num_workers=num_workers)
    valloader_cls_s = DataLoader(dataset=valset_cls_s, batch_size=config['test_size'], shuffle=False, num_workers=num_workers)

    # Target domain dataset
    transform = transform_setup(config, "classification", mode="train")
    trainset_cls_t = ClassificationDataset(config['data_root_cls'], config['data_list_t']['train'], transform=transform, domain='target')
    transform = transform_setup(config, "classification", mode="val")
    valset_cls_t = ClassificationDataset(config['data_root_cls'], config['data_list_t']['val'], transform=transform, domain='target')
    # Dataloader
    trainloader_cls_t = DataLoader(dataset=trainset_cls_t, batch_size=config['cls_batch_size'], shuffle=True, num_workers=num_workers)
    valloader_cls_t = DataLoader(dataset=valset_cls_t, batch_size=config['test_size'], shuffle=False, num_workers=num_workers)

    print("\nClassification")
    print(f"Num(Source): ({len(trainset_cls_s)}, {len(valset_cls_s)})", end=' | ')
    print(f"Num(Target): ({len(trainset_cls_t)}, {len(valset_cls_t)})", end=' | ')
    print(f"Num(batch): {len(trainloader_cls_s)}/epoch\n")

########################################
# Segmentation data loader
########################################
if config['segmentor'] is not None:
    # Source domain dataset
    transform = transform_setup(config, "segmentation", mode="train")
    trainset_seg_s = GTA5Dataset(config['data_root_seg'], config['data_list']['train'], transform=transform, mode='train')
    transform = None
    valset_seg_s = GTA5Dataset(config['data_root_seg'], config['data_list']['val'], transform=transform, mode='val')
    # Dataloader
    trainloader_seg_s = DataLoader(dataset=trainset_seg_s, batch_size=config['seg_batch_size'], shuffle=True, num_workers=num_workers)
    valloader_seg_s = DataLoader(dataset=valset_seg_s, batch_size=config['test_size'], shuffle=False, num_workers=num_workers)

    # Target domain dataset
    transform = transform_setup(config, "segmentation", mode="train")
    trainset_seg_t = CityScapesDataset(config['data_root_seg'], config['data_list']['train'], transform=transform, mode='train')
    transform = None
    valset_seg_t = CityScapesDataset(config['data_root_seg'], config['data_list']['test'], transform=transform, mode='test')
    # Dataloader
    trainloader_seg_t = DataLoader(dataset=trainset_seg_t, batch_size=config['seg_batch_size'], shuffle=True, num_workers=num_workers)
    valloader_seg_t = DataLoader(dataset=valset_seg_t, batch_size=config['test_size'], shuffle=False, num_workers=num_workers)

    print("Segmentation dataset")
    print(f"Num(Source): ({len(trainset_seg_s)}, {len(valset_seg_s)})", end=' | ')
    print(f"Num(Target): ({len(trainset_seg_t)}, {len(valset_seg_t)})", end=' | ')
    print(f"Num(batch): {len(trainloader_seg_s)}/epoch\n")

# Data loader list
if config['task'] == "Classification":
    dataloader_list = [trainloader_cls_s, valloader_cls_s, trainloader_cls_t, valloader_cls_t]
elif config['task'] == "Segmentation":
    dataloader_list = [trainloader_seg_s, valloader_seg_s, trainloader_seg_t, valloader_seg_t]
elif config['task'] == "Multi_task":
    dataloader_list = [trainloader_cls_s, valloader_cls_s, trainloader_cls_t, valloader_cls_t,
                       trainloader_seg_s, valloader_seg_s, trainloader_seg_t, valloader_seg_t]
else:
    raise RuntimeError("Task doesn't support.")

########################################
# Model
########################################
if config['trainer'] == 'Classify':
    trainer = Classify(config)
elif config['trainer'] == 'ClassifyAdapt':
    trainer = ClassifyAdapt(config)
elif config['trainer'] == 'Segment':
    trainer = Segment(config)
elif config['trainer'] == 'SegmentAdapt':
    trainer = SegmentAdapt(config)
elif config['trainer'] == 'Multi_task':
    trainer = Multi_task(config)
elif config['trainer'] == 'MTAdapt':
    trainer = MTAdapt(config)
trainer.cuda()


# Only for translate (Reduce GPU memory)
if len(config['freeze_list']) >= 5:
    trainer.enc_bridge.cpu()
    trainer.dis_h.cpu()
    if config['task'] == 'Segmentation':
        trainer.segmentor.cpu()
    if config['task'] == 'Classification':
        trainer.classifier.cpu()

########################################
# Start training
########################################
if config['trainer'] == 'Classify':
    train_classification(dataloader_list)
elif config['trainer'] == 'Segment':
    train_segmentation(dataloader_list)
else:
    train_multi_task_adapt(dataloader_list)
