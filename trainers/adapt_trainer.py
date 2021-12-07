from os.path import join as PJ
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from networks.resnet_based import FCN8sSegment, ResDis, ResNetClassifier
from networks.resnet_based import ResNet101EncoderBridge
from networks.resnet_based import Res101GenShare, Res101Gen
from utils import get_model_list, get_scheduler


###############################################################################
# Adaptation
###############################################################################
class SingleBasicAdapt(nn.Module):
    """ Single task adaptation, pretrained source only """
    def __init__(self, config):
        super().__init__()
        self._initial_networks(config)
        self._initial_optimizer(config)
        self.num_class = config['num_class']

    def _initial_networks(self, config):
        pass

    def _freeze_batch_norm(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.affine = False
                m.track_running_stats = False

    def _freeze(self, freeze_list):
        freeze_param = []
        print("Freeze:", end=" ")
        # Add model into freeze_param list
        if 'gen_a_enc' in freeze_list:
            freeze_param += list(self.gen_a.enc.parameters())
            self._freeze_batch_norm(self.gen_a.enc)
            print("[gen_a.enc]", end=" | ")

        if 'classifier' in freeze_list:
            freeze_param += list(self.classifier.parameters())
            print("[classifier]", end=" | ")

        if 'segmentor' in freeze_list:
            freeze_param += list(self.segmentor.parameters())
            print("[segmentor]", end=" | ")

        if 'gen_b_enc' in freeze_list:
            freeze_param += list(self.gen_b.enc.parameters())
            print("[gen_b.enc]", end=" | ")
            self._freeze_batch_norm(self.gen_b.enc)

        if 'enc_bridge' in freeze_list:
            freeze_param += list(self.enc_bridge.parameters())
            print("[enc_bridge]", end=" | ")
            self._freeze_batch_norm(self.enc_bridge)

        if 'dis_h' in freeze_list:
            freeze_param += list(self.dis_h.parameters())
            print("[dis_h]", end=" | ")
        print("\n")

        # Freeze parameters
        for param in freeze_param:
            param.requires_grad = False

    def _initial_optimizer(self, config):
        pass

    #################################################################
    # Forward
    #################################################################
    def forward(self, x_a, x_b):
        """ Trnaslation forward """
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b, x_a.shape)
        x_ab = self.gen_b.decode(h_a, x_b.shape)
        self.train()
        return x_ab, x_ba

    def classify(self, x, domain):
        """ Classification forward """
        self.eval()
        gen = self.gen_a if domain == "source" else self.gen_b
        h, _ = gen.encode(x, seg=False)
        h = self.enc_bridge(h, seg=False)
        # predicts = self.classifier(h).type(torch.float)
        # self.train()
        # return predicts

        predicts, features = self.classifier(h)
        predicts = predicts.type(torch.float)
        self.train()
        return predicts, features

    def segment(self, x, domain):
        """ Segmetation forward """
        self.eval()
        gen = self.gen_a if domain == "source" else self.gen_b
        h, _ = gen.encode(x, seg=True)
        pool3, pool4, pool5 = self.enc_bridge(h, seg=True)
        predicts = self.segmentor(pool3, pool4, pool5, x.shape).type(torch.float)
        predicts = torch.argmax(predicts, dim=1, keepdim=True)
        self.train()
        return predicts

    ##################################################
    # Calculate loss
    ##################################################
    def recon_criterion(self, x, targets):
        return torch.mean(torch.abs(x - targets))

    def compute_classify_loss(self, h, targets):
        targets = targets.view(-1)
        predicts = self.classifier(h).type(torch.float)
        criterion = nn.CrossEntropyLoss()
        return criterion(predicts, targets)

    def compute_segmentation_loss(self, pool3, pool4, pool5, x_shape, targets):
        predicts = self.segmentor(pool3, pool4, pool5, x_shape).type(torch.float)
        predicts = nn.LogSoftmax(dim=1)(predicts)
        criterion = torch.nn.NLLLoss(ignore_index=19)
        loss = criterion(predicts, targets)
        return loss

    #################################################################
    # Training
    #################################################################
    def gen_update(self, x_a, x_b, targets, config):
        pass

    def dis_update(self, x_a, x_b, config):
        # Initialize
        t_shape_a, t_shape_b = x_a.shape, x_b.shape
        self.loss_dis_total = 0
        self.dis_opt.zero_grad()

        # Encode to latent space
        h_a, n_a = self.gen_a.encode(x_a, seg=False)
        h_b, n_b = self.gen_b.encode(x_b, seg=False)

        # Decode to other domain (Translate)
        x_ab = self.gen_b.decode(h_a + n_a, t_shape_b)
        x_ba = self.gen_a.decode(h_b + n_b, t_shape_a)

        # [Q_TD] Translate discriminator loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)

        # [Q_LD] Latent discriminator loss
        self.loss_dis_h = self.dis_h.calc_dis_loss(h_b.detach(), h_a.detach()) if config['latent_dis_w'] else 0

        # Total loss
        self.loss_dis_total =\
            config['tran_dis_w'] * self.loss_dis_a +\
            config['tran_dis_w'] * self.loss_dis_b +\
            config['latent_dis_w'] * self.loss_dis_h

        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    ################################################################################
    # Saving and loading model
    ################################################################################
    def save(self, config, snapshot_dir, iterations):
        """ Save generators, discriminators, classifier/segmentor and optimizers """
        gen_name = PJ(snapshot_dir, f'gen_{iterations:08d}.pt')
        bdg_name = PJ(snapshot_dir, f'bdg_{iterations:08d}.pt')
        dis_name = PJ(snapshot_dir, f'dis_{iterations:08d}.pt')
        cls_name = PJ(snapshot_dir, f'cls_{iterations:08d}.pt')
        seg_name = PJ(snapshot_dir, f'seg_{iterations:08d}.pt')
        opt_name = PJ(snapshot_dir, 'optimizer.pt')

        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict(), 'h': self.dis_h.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

        if config['classifier']:
            torch.save(self.classifier.state_dict(), cls_name)
        if config['segmentor']:
            torch.save(self.segmentor.state_dict(), seg_name)
        if config['classifier'] or config['segmentor']:
            torch.save(self.enc_bridge.state_dict(), bdg_name)
        print(f"Saving models in {snapshot_dir} finished.")

    def resume(self, config, checkpoint_dir):
        """ Resume training training progress. """
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])

        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        self.dis_h.load_state_dict(state_dict['h'])

        # Load Classifier
        if config['classifier']:
            last_model_name = get_model_list(checkpoint_dir, "cls")
            state_dict = torch.load(last_model_name)
            self.classifier.load_state_dict(state_dict)
        # Load Segmentor
        if config['segmentor']:
            last_model_name = get_model_list(checkpoint_dir, "seg")
            state_dict = torch.load(last_model_name)
            self.segmentor.load_state_dict(state_dict)
        # Load encoder bridge
        if config['classifier'] or config['segmentor']:
            last_model_name = get_model_list(checkpoint_dir, "bdg")
            state_dict = torch.load(last_model_name)
            self.enc_bridge.load_state_dict(state_dict)

        # Load optimizers
        state_dict = torch.load(PJ(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, config, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, config, iterations)
        print(f'Resume from iteration {iterations}')
        return iterations

    def load_weights(self, checkpoint_dir, gen_weight=None, dis_weight=None, cls_weight=None, seg_weight=None, bdg_weight=None):
        """ Load trained model. (almost equal _pretrained)"""
        print(f"Loading model weights:")
        # Load generator
        if gen_weight:
            state_dict = torch.load(PJ(checkpoint_dir, gen_weight))
            self.gen_a.load_state_dict(state_dict['a'])
            self.gen_b.load_state_dict(state_dict['b'])
            print(f"{gen_weight}", end="  ")
        # Load dicriminator
        if dis_weight:
            state_dict = torch.load(PJ(checkpoint_dir, dis_weight))
            self.dis_a.load_state_dict(state_dict['a'])
            self.dis_b.load_state_dict(state_dict['b'])
            self.dis_h.load_state_dict(state_dict['h'])
            print(f"{dis_weight}", end="  ")
        # Load classifier
        if cls_weight:
            state_dict = torch.load(PJ(checkpoint_dir, cls_weight))
            self.classifier.load_state_dict(state_dict)
            print(f"{cls_weight}", end="  ")
        # Load Segmentor
        if seg_weight:
            state_dict = torch.load(PJ(checkpoint_dir, seg_weight))
            self.segmentor.load_state_dict(state_dict)
            print(f"{seg_weight}", end="  ")
        # Load encoder bridge
        if bdg_weight:
            state_dict = torch.load(PJ(checkpoint_dir, bdg_weight))
            self.enc_bridge.load_state_dict(state_dict)
            print(f"{bdg_weight}", end="  ")
        print("finished.\n")

    def _pretrained(self, config):
        """ Load pretrained model. """
        print("Load pretrained model:")
        if config['gen_weights']:
            enc_name = PJ(config['pretrained_checkpoint'], config['gen_weights'])
            state_dict = torch.load(enc_name)
            self.gen_a.load_state_dict(state_dict['a'])
            self.gen_b.load_state_dict(state_dict['b'])
            print(f"{config['gen_weights']} ", end="  ")
        if config['enc_weights']:
            enc_name = PJ(config['pretrained_checkpoint'], config['enc_weights'])
            state_dict = torch.load(enc_name)
            self.gen_a.enc.load_state_dict(state_dict)
            self.gen_b.enc.load_state_dict(state_dict)
            print(f"{config['enc_weights']} ", end="  ")
        if config['dis_weights']:
            enc_name = PJ(config['pretrained_checkpoint'], config['dis_weights'])
            state_dict = torch.load(enc_name)
            self.dis_a.load_state_dict(state_dict['a'])
            self.dis_b.load_state_dict(state_dict['b'])
            self.dis_h.load_state_dict(state_dict['h'])
            print(f"{config['dis_weights']} ", end="  ")
        if config['bdg_weights']:
            bdg_name = PJ(config['pretrained_checkpoint'], config['bdg_weights'])
            state_dict = torch.load(bdg_name)
            self.enc_bridge.load_state_dict(state_dict)
            print(f"{config['bdg_weights']} ", end="  ")
        if config['cls_weights']:
            cls_name = PJ(config['pretrained_checkpoint'], config['cls_weights'])
            state_dict = torch.load(cls_name)
            self.classifier.load_state_dict(state_dict)
            print(f"{config['cls_weights']} ", end="  ")
        if config['seg_weights']:
            seg_name = PJ(config['pretrained_checkpoint'], config['seg_weights'])
            state_dict = torch.load(seg_name)
            self.segmentor.load_state_dict(state_dict)
            print(f"{config['seg_weights']} ", end="  ")
        print(f"\nLoad pretrained model from {config['pretrained_checkpoint']}")

    #################################################################
    # Record images
    #################################################################
    def sample(self, x_a, x_b):
        self.eval()
        t_shape_a, t_shape_b = x_a.shape, x_b.shape
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0), seg=False)
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0), seg=False)
            x_a_recon.append(self.gen_a.decode(h_a, t_shape_a))
            x_b_recon.append(self.gen_b.decode(h_b, t_shape_b))
            x_ba.append(self.gen_a.decode(h_b, t_shape_a))
            x_ab.append(self.gen_b.decode(h_a, t_shape_b))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba, x_ab = torch.cat(x_ba), torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def record_translate_images(self, x_a, x_b, image_directory, name, _scale=0.5):
        num_image, _, h, w = x_a.shape
        mean = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)

        x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba = self.sample(x_a, x_b)

        # Denormalize
        visual_a = torch.cat([x_a, x_a_recon, x_ab], dim=0).cpu()
        visual_b = torch.cat([x_b, x_b_recon, x_ba], dim=0).cpu()
        visual_a = visual_a * std + mean
        visual_b = visual_b * std + mean
        # Resize
        visual_a = nn.functional.interpolate(visual_a, scale_factor=_scale)
        visual_b = nn.functional.interpolate(visual_b, scale_factor=_scale)
        # Combine
        visual_a = vutils.make_grid(visual_a, nrow=num_image, padding=2, scale_each=True)
        visual_b = vutils.make_grid(visual_b, nrow=num_image, padding=2, scale_each=True)
        # Save results
        vutils.save_image(visual_a, PJ(image_directory, f"{name}_a.jpg"), nrow=1)
        vutils.save_image(visual_b, PJ(image_directory, f"{name}_b.jpg"), nrow=1)
        x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba = None, None, None, None, None, None

    def record_segment_images(self, labels, images, segment_directory, name, results=None, domain=None, _scale=0.5):
        # Skip background
        num_class = self.num_class['seg'] - 1
        num_image, h, w = labels.shape

        labels = labels.cpu().unsqueeze(1)
        results = self.segment(images, domain).cpu() if results is None else results
        labels_visual = torch.zeros((num_image, 3, h, w), dtype=torch.float32)
        results_visual = torch.zeros((num_image, 3, h, w), dtype=torch.float32)

        # Replace background pixel in results by 19
        mask_background = (labels == 19)
        results[mask_background] = 19.

        # Draw color (Categories)
        for i in range(num_class):
            color = self.id_to_color[i].reshape(1, 3, 1, 1)
            # Label (using mask which size is equal to _visual)
            l_mask = (labels == i).repeat(1, 3, 1, 1).type(torch.float32)
            labels_visual += l_mask * color
            # Result (using mask which size is equal to _visual)
            r_mask = (results == i).repeat(1, 3, 1, 1).type(torch.float32)
            results_visual += r_mask * color

        # Denormalize
        mean = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        images = images.cpu() * std + mean
        labels_visual = labels_visual / 255.
        results_visual = results_visual / 255.
        # Save original images, labels, results
        visual = torch.cat([images, labels_visual, results_visual], dim=2)
        visual = nn.functional.interpolate(visual, scale_factor=_scale)
        num_image = 10 if num_image > 10 else num_image
        image_grid = vutils.make_grid(visual, nrow=num_image, padding=2, scale_each=True)
        vutils.save_image(image_grid, PJ(segment_directory, f"{name}.jpg"), nrow=1)


class ClassifyAdapt(SingleBasicAdapt):
    def __init__(self, config):
        super().__init__(config)

    def _initial_networks(self, config):
        init = config['init']
        # Create networks
        if config['classifier'] == 'ResNet101':
            self.gen_a = Res101Gen(init)
            self.gen_b = Res101Gen(init)
            self.dis_a = ResDis(3, config['dis'], init)
            self.dis_b = ResDis(3, config['dis'], init)
            self.dis_h = ResDis(256, config['dis_h'], init)
            self.enc_bridge = ResNet101EncoderBridge()
            self.classifier = ResNetClassifier(config['num_class']['cls'], init)

        self._freeze_batch_norm(self.enc_bridge)
        self._freeze_batch_norm(self.gen_b)
        if config['pretrained_checkpoint']:
            self._pretrained(config)
            self._freeze(config['freeze_list'])

    def _freeze_batch_norm(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # m.affine = False
                # m.track_running_stats = False
                m.affine = True
                m.track_running_stats = True

    def _initial_optimizer(self, config):
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters()) + list(self.dis_h.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters()) + list(self.enc_bridge.parameters()) + list(self.classifier.parameters())
        self.dis_opt = torch.optim.SGD([p for p in dis_params if p.requires_grad],
                                       lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
        self.gen_opt = torch.optim.SGD([p for p in gen_params if p.requires_grad],
                                       lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, config)
        self.gen_scheduler = get_scheduler(self.gen_opt, config)

    #################################################################
    # Training
    #################################################################
    def gen_update(self, x_a, x_b, targets, config):
        # Initialize
        t_shape_a, t_shape_b = x_a.shape, x_b.shape
        self.loss_gen_total = 0
        self.gen_opt.zero_grad()

        # Encode to latent space
        h_a, n_a = self.gen_a.encode(x_a, seg=False)
        h_b, n_b = self.gen_b.encode(x_b, seg=False)

        # Decode to image space (Reconstruct)
        x_a_recon = self.gen_a.decode(h_a + n_a, t_shape_a)
        x_b_recon = self.gen_b.decode(h_b + n_b, t_shape_b)

        # Decode to other domain (Translate)
        x_ab = self.gen_b.decode(h_a + n_a, t_shape_b)
        x_ba = self.gen_a.decode(h_b + n_b, t_shape_a)

        # Encode to latent again
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab, seg=False)
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba, seg=False)

        # Decode to original space (Cycle reconstruct)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon, t_shape_a)
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon, t_shape_b)

        # [Q_cyc] Cycle reconstruction loss
        self.loss_gen_cyc_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_b = self.recon_criterion(x_bab, x_b)

        # [Q_R] Reconstruction loss
        self.loss_gen_recon_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_b = self.recon_criterion(x_b_recon, x_b)

        # [Q_LG] Latent generator adversarial loss
        self.loss_gen_adv_h = self.dis_h.calc_gen_loss(h_b) if config['latent_dis_w'] else 0
        # [Q_TG] Translation generator loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        # [Q_S] Supervised loss
        self.loss_cls = 0
        if config['cls_w']:
            h_a = self.enc_bridge(h_a, seg=False)
            self.loss_cls = self.compute_classify_loss(h_a, targets)

        # [Q_TS] Supervised loss
        self.loss_tran_cls = 0
        if config['tran_cls_w']:
            h_a_recon = self.enc_bridge(h_a_recon, seg=False)
            self.loss_tran_cls = self.compute_classify_loss(h_a_recon, targets)

        self.loss_gen_total =\
            config['cyc_w'] * self.loss_gen_cyc_a +\
            config['cyc_w'] * self.loss_gen_cyc_b +\
            config['recon_w'] * self.loss_gen_recon_a +\
            config['recon_w'] * self.loss_gen_recon_b +\
            config['latent_gen_w'] * self.loss_gen_adv_h +\
            config['tran_gen_w'] * self.loss_gen_adv_a +\
            config['tran_gen_w'] * self.loss_gen_adv_b +\
            config['cls_w'] * self.loss_cls +\
            config['tran_cls_w'] * self.loss_tran_cls

        self.loss_gen_total.backward()
        self.gen_opt.step()


class SegmentAdapt(SingleBasicAdapt):
    def __init__(self, config):
        super().__init__(config)

        self.id_to_color = torch.tensor(
            [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
             [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
             [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
             [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]], dtype=torch.float32)

    def _initial_networks(self, config):
        init = config['init']
        # Create networks
        if config['segmentor'] == 'ResNet101':
            self.gen_a = Res101Gen(init)
            self.gen_b = Res101Gen(init)
            self.dis_a = ResDis(3, config['dis'], init)
            self.dis_b = ResDis(3, config['dis'], init)
            self.dis_h = ResDis(256, config['dis_h'], init)
            self.enc_bridge = ResNet101EncoderBridge()
            self.segmentor = FCN8sSegment(config['num_class']['seg'], init)

        if config['pretrained_checkpoint']:
            self._pretrained(config)
            self._freeze(config['freeze_list'])

    def _initial_optimizer(self, config):
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters()) + list(self.dis_h.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters()) + list(self.enc_bridge.parameters()) + list(self.segmentor.parameters())
        self.dis_opt = torch.optim.SGD([p for p in dis_params if p.requires_grad],
                                       lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
        self.gen_opt = torch.optim.SGD([p for p in gen_params if p.requires_grad],
                                       lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, config)
        self.gen_scheduler = get_scheduler(self.gen_opt, config)

    #################################################################
    # Training
    #################################################################
    def gen_update(self, x_a, x_b, targets, config):
        # Initialize
        t_shape_a, t_shape_b = x_a.shape, x_b.shape
        self.loss_gen_total = 0
        self.gen_opt.zero_grad()

        # Encode to latent space
        h_a, n_a = self.gen_a.encode(x_a, seg=False)
        h_b, n_b = self.gen_b.encode(x_b, seg=False)

        # Decode to image space (Reconstruct)
        x_a_recon = self.gen_a.decode(h_a + n_a, t_shape_a)
        x_b_recon = self.gen_b.decode(h_b + n_b, t_shape_b)

        # Decode to other domain (Translate)
        x_ab = self.gen_b.decode(h_a + n_a, t_shape_b)
        x_ba = self.gen_a.decode(h_b + n_b, t_shape_a)

        self.loss_gen_cyc_a, self.loss_gen_cyc_b = 0, 0
        if config['cyc_w']:
            # Encode to latent again
            h_a_recon, n_a_recon = self.gen_b.encode(x_ab, seg=False)
            h_b_recon, n_b_recon = self.gen_a.encode(x_ba, seg=False)

            # Decode to original space (Cycle reconstruct)
            x_aba = self.gen_a.decode(h_a_recon + n_a_recon, t_shape_a)
            x_bab = self.gen_b.decode(h_b_recon + n_b_recon, t_shape_b)

            # [Q_cyc] Cycle reconstruction loss
            self.loss_gen_cyc_a = self.recon_criterion(x_aba, x_a)
            self.loss_gen_cyc_b = self.recon_criterion(x_bab, x_b)

        # [Q_R] Reconstruction loss
        self.loss_gen_recon_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_b = self.recon_criterion(x_b_recon, x_b)

        # [Q_LG] Latent generator adversarial loss
        self.loss_gen_adv_h = self.dis_h.calc_gen_loss(h_b) if config['latent_gen_w'] else 0
        # [Q_TG] Translation generator loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        # [Q_S] Supervised loss
        self.loss_seg = 0
        if config['seg_w']:
            h_a, _ = self.gen_a.encode(x_a, seg=True)
            pool3, pool4, pool5 = self.enc_bridge(h_a, seg=True)
            self.loss_seg = self.compute_segmentation_loss(pool3, pool4, pool5, t_shape_a, targets)

        # [Q_TS] Supervised loss
        self.loss_tran_seg = 0
        if config['tran_seg_w']:
            h_a_recon, _ = self.gen_b.encode(x_ab, seg=True)
            pool3_recon, pool4_recon, pool5_recon = self.enc_bridge(h_a_recon, seg=True)
            self.loss_tran_seg = self.compute_segmentation_loss(pool3_recon, pool4_recon, pool5_recon, t_shape_a, targets)

        self.loss_gen_total =\
            config['cyc_w'] * self.loss_gen_cyc_a +\
            config['cyc_w'] * self.loss_gen_cyc_b +\
            config['recon_w'] * self.loss_gen_recon_a +\
            config['recon_w'] * self.loss_gen_recon_b +\
            config['latent_gen_w'] * self.loss_gen_adv_h +\
            config['tran_gen_w'] * self.loss_gen_adv_a +\
            config['tran_gen_w'] * self.loss_gen_adv_b +\
            config['seg_w'] * self.loss_seg +\
            config['tran_seg_w'] * self.loss_tran_seg

        self.loss_gen_total.backward()
        self.gen_opt.step()


class BasicAdapt(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._initial_networks(config)
        self._initial_optimizer(config)
        self.num_class = config['num_class']

    def _initial_networks(self, config):
        pass

    def _freeze_batch_norm(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.affine = False
                m.track_running_stats = False

    def _freeze(self, freeze_list):
        freeze_param = []
        print("Freeze:", end=" ")
        # Add model into freeze_param list
        if 'gen_a_enc' in freeze_list:
            freeze_param += list(self.gen_a.enc.parameters())
            self._freeze_batch_norm(self.gen_a.enc)
            print("[gen_a.enc]", end=" | ")

        if 'classifier' in freeze_list:
            freeze_param += list(self.classifier.parameters())
            print("[classifier]", end=" | ")

        if 'segmentor' in freeze_list:
            freeze_param += list(self.segmentor.parameters())
            print("[segmentor]", end=" | ")

        if 'gen_b_enc' in freeze_list:
            freeze_param += list(self.gen_b.enc.parameters())
            print("[gen_b.enc]", end=" | ")
            self._freeze_batch_norm(self.gen_b.enc)

        if 'enc_bridge' in freeze_list:
            freeze_param += list(self.enc_bridge.parameters())
            print("[enc_bridge]", end=" | ")
            self._freeze_batch_norm(self.enc_bridge)

        if 'dis_h' in freeze_list:
            freeze_param += list(self.dis_h.parameters())
            print("[dis_h]", end=" | ")
        print("\n")

        # Freeze parameters
        for param in freeze_param:
            param.requires_grad = False

    def _initial_optimizer(self, config):
        pass

    #################################################################
    # Forward
    #################################################################
    def classify(self, x, domain):
        """ Classification forward """
        self.eval()
        gen = self.gen_a if domain == "source" else self.gen_b
        h, _ = gen.encode(x, seg=False)
        h = self.enc_bridge(h, seg=False)
        predicts = self.classifier(h).type(torch.float)
        self.train()
        return predicts

    def segment(self, x, domain):
        """ Segmetation forward """
        self.eval()
        gen = self.gen_a if domain == "source" else self.gen_b
        h, _ = gen.encode(x, seg=True)
        pool3, pool4, pool5 = self.enc_bridge(h, seg=True)
        predicts = self.segmentor(pool3, pool4, pool5, x.shape).type(torch.float)
        predicts = torch.argmax(predicts, dim=1, keepdim=True)
        self.train()
        return predicts

    ##################################################
    # Calculate loss
    ##################################################
    def recon_criterion(self, x, targets):
        return torch.mean(torch.abs(x - targets))

    def compute_classify_loss(self, h, targets):
        targets = targets.view(-1)
        predicts = self.classifier(h).type(torch.float)
        criterion = nn.CrossEntropyLoss()
        return criterion(predicts, targets)

    def compute_segmentation_loss(self, pool3, pool4, pool5, x_shape, targets):
        predicts = self.segmentor(pool3, pool4, pool5, x_shape).type(torch.float)
        predicts = nn.LogSoftmax(dim=1)(predicts)
        criterion = torch.nn.NLLLoss(ignore_index=19)
        loss = criterion(predicts, targets)
        return loss

    #################################################################
    # Training
    #################################################################
    def gen_update(self, x_a, x_b, targets, config):
        pass

    def dis_update(self, x_a, x_b, config):
        # Initialize
        t_shape_a, t_shape_b = x_a.shape, x_b.shape
        self.loss_dis_total = 0
        self.dis_opt.zero_grad()

        # Encode to latent space
        h_a, n_a = self.gen_a.encode(x_a, seg=False)
        h_b, n_b = self.gen_b.encode(x_b, seg=False)

        # Decode to other domain (Translate)
        x_ab = self.gen_b.decode(h_a + n_a, t_shape_b)
        x_ba = self.gen_a.decode(h_b + n_b, t_shape_a)

        # [Q_TD] Translate discriminator loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)

        # [Q_LD] Latent discriminator loss
        self.loss_dis_h = self.dis_h.calc_dis_loss(h_b.detach(), h_a.detach()) if config['latent_dis_w'] else 0

        # Total loss
        self.loss_dis_total =\
            config['tran_dis_w'] * self.loss_dis_a +\
            config['tran_dis_w'] * self.loss_dis_b +\
            config['latent_dis_w'] * self.loss_dis_h

        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    ################################################################################
    # Saving and loading model
    ################################################################################
    def save(self, config, snapshot_dir, iterations):
        """ Save generators, discriminators, classifier/segmentor and optimizers """
        gen_name = PJ(snapshot_dir, f'gen_{iterations:08d}.pt')
        bdg_name = PJ(snapshot_dir, f'bdg_{iterations:08d}.pt')
        dis_name = PJ(snapshot_dir, f'dis_{iterations:08d}.pt')
        cls_name = PJ(snapshot_dir, f'cls_{iterations:08d}.pt')
        seg_name = PJ(snapshot_dir, f'seg_{iterations:08d}.pt')
        opt_name = PJ(snapshot_dir, 'optimizer.pt')

        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict(), 'h': self.dis_h.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

        if config['classifier']:
            torch.save(self.classifier.state_dict(), cls_name)
        if config['segmentor']:
            torch.save(self.segmentor.state_dict(), seg_name)
        if config['classifier'] or config['segmentor']:
            torch.save(self.enc_bridge.state_dict(), bdg_name)
        print(f"Saving models in {snapshot_dir} finished.")

    def resume(self, config, checkpoint_dir):
        """ Resume training training progress. """
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])

        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        self.dis_h.load_state_dict(state_dict['h'])

        # Load Classifier
        if config['classifier']:
            last_model_name = get_model_list(checkpoint_dir, "cls")
            state_dict = torch.load(last_model_name)
            self.classifier.load_state_dict(state_dict)
        # Load Segmentor
        if config['segmentor']:
            last_model_name = get_model_list(checkpoint_dir, "seg")
            state_dict = torch.load(last_model_name)
            self.segmentor.load_state_dict(state_dict)
        # Load encoder bridge
        if config['classifier'] or config['segmentor']:
            last_model_name = get_model_list(checkpoint_dir, "bdg")
            state_dict = torch.load(last_model_name)
            self.enc_bridge.load_state_dict(state_dict)

        # Load optimizers
        state_dict = torch.load(PJ(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, config, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, config, iterations)
        print(f'Resume from iteration {iterations}')
        return iterations

    def load_weights(self, checkpoint_dir, gen_weight=None, dis_weight=None, cls_weight=None, seg_weight=None, bdg_weight=None):
        """ Load trained model. (almost equal _pretrained)"""
        print(f"Loading model weights:")
        # Load generator
        if gen_weight:
            state_dict = torch.load(PJ(checkpoint_dir, gen_weight))
            self.gen_a.load_state_dict(state_dict['a'])
            self.gen_b.load_state_dict(state_dict['b'])
            print(f"{gen_weight}", end="  ")
        # Load dicriminator
        if dis_weight:
            state_dict = torch.load(PJ(checkpoint_dir, dis_weight))
            self.dis_a.load_state_dict(state_dict['a'])
            self.dis_b.load_state_dict(state_dict['b'])
            self.dis_h.load_state_dict(state_dict['h'])
            print(f"{dis_weight}", end="  ")
        # Load classifier
        if cls_weight:
            state_dict = torch.load(PJ(checkpoint_dir, cls_weight))
            self.classifier.load_state_dict(state_dict)
            print(f"{cls_weight}", end="  ")
        # Load Segmentor
        if seg_weight:
            state_dict = torch.load(PJ(checkpoint_dir, seg_weight))
            self.segmentor.load_state_dict(state_dict)
            print(f"{seg_weight}", end="  ")
        # Load encoder bridge
        if bdg_weight:
            state_dict = torch.load(PJ(checkpoint_dir, bdg_weight))
            self.enc_bridge.load_state_dict(state_dict)
            print(f"{bdg_weight}", end="  ")
        print("finished.\n")

    def _pretrained(self, config):
        """ Load pretrained model. """
        print("Load pretrained model:")
        if config['gen_weights']:
            enc_name = PJ(config['pretrained_checkpoint'], config['gen_weights'])
            state_dict = torch.load(enc_name)
            self.gen_a.load_state_dict(state_dict['a'])
            self.gen_b.load_state_dict(state_dict['b'])
            print(f"{config['gen_weights']} ", end="  ")
        if config['enc_weights']:
            enc_name = PJ(config['pretrained_checkpoint'], config['enc_weights'])
            state_dict = torch.load(enc_name)
            self.gen_a.enc.load_state_dict(state_dict)
            self.gen_b.enc.load_state_dict(state_dict)
            print(f"{config['enc_weights']} ", end="  ")
        if config['dis_weights']:
            enc_name = PJ(config['pretrained_checkpoint'], config['dis_weights'])
            state_dict = torch.load(enc_name)
            self.dis_a.load_state_dict(state_dict['a'])
            self.dis_b.load_state_dict(state_dict['b'])
            self.dis_h.load_state_dict(state_dict['h'])
            print(f"{config['dis_weights']} ", end="  ")
        if config['bdg_weights']:
            bdg_name = PJ(config['pretrained_checkpoint'], config['bdg_weights'])
            state_dict = torch.load(bdg_name)
            self.enc_bridge.load_state_dict(state_dict)
            print(f"{config['bdg_weights']} ", end="  ")
        if config['cls_weights']:
            cls_name = PJ(config['pretrained_checkpoint'], config['cls_weights'])
            state_dict = torch.load(cls_name)
            self.classifier.load_state_dict(state_dict)
            print(f"{config['cls_weights']} ", end="  ")
        if config['seg_weights']:
            seg_name = PJ(config['pretrained_checkpoint'], config['seg_weights'])
            state_dict = torch.load(seg_name)
            self.segmentor.load_state_dict(state_dict)
            print(f"{config['seg_weights']} ", end="  ")
        print(f"\nLoad pretrained model from {config['pretrained_checkpoint']}")

    #################################################################
    # Record images
    #################################################################
    def sample(self, x_a, x_b):
        self.eval()
        t_shape_a, t_shape_b = x_a.shape, x_b.shape
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0), seg=False)
            h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0), seg=False)
            x_a_recon.append(self.gen_a.decode(h_a, t_shape_a))
            x_b_recon.append(self.gen_b.decode(h_b, t_shape_b))
            x_ba.append(self.gen_a.decode(h_b, t_shape_a))
            x_ab.append(self.gen_b.decode(h_a, t_shape_b))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba, x_ab = torch.cat(x_ba), torch.cat(x_ab)
        self.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def record_translate_images(self, x_a, x_b, image_directory, name, _scale=0.5):
        num_image, _, h, w = x_a.shape
        mean = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)

        x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba = self.sample(x_a, x_b)

        # Denormalize
        visual_a = torch.cat([x_a, x_a_recon, x_ab], dim=0).cpu()
        visual_b = torch.cat([x_b, x_b_recon, x_ba], dim=0).cpu()
        visual_a = visual_a * std + mean
        visual_b = visual_b * std + mean
        # Resize
        visual_a = nn.functional.interpolate(visual_a, scale_factor=_scale)
        visual_b = nn.functional.interpolate(visual_b, scale_factor=_scale)
        # Combine
        visual_a = vutils.make_grid(visual_a, nrow=num_image, padding=2, scale_each=True)
        visual_b = vutils.make_grid(visual_b, nrow=num_image, padding=2, scale_each=True)
        # Save results
        vutils.save_image(visual_a, PJ(image_directory, f"{name}_a.jpg"), nrow=1)
        vutils.save_image(visual_b, PJ(image_directory, f"{name}_b.jpg"), nrow=1)
        x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba = None, None, None, None, None, None

    def record_segment_images(self, labels, images, segment_directory, name, results=None, domain=None, _scale=0.5):
        # Skip background
        num_class = self.num_class['seg'] - 1
        num_image, h, w = labels.shape

        labels = labels.cpu().unsqueeze(1)
        results = self.segment(images, domain).cpu() if results is None else results
        labels_visual = torch.zeros((num_image, 3, h, w), dtype=torch.float32)
        results_visual = torch.zeros((num_image, 3, h, w), dtype=torch.float32)

        # Replace background pixel in results by 19
        mask_background = (labels == 19)
        results[mask_background] = 19.

        # Draw color (Categories)
        for i in range(num_class):
            color = self.id_to_color[i].reshape(1, 3, 1, 1)
            # Label (using mask which size is equal to _visual)
            l_mask = (labels == i).repeat(1, 3, 1, 1).type(torch.float32)
            labels_visual += l_mask * color
            # Result (using mask which size is equal to _visual)
            r_mask = (results == i).repeat(1, 3, 1, 1).type(torch.float32)
            results_visual += r_mask * color

        # Denormalize
        mean = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        images = images.cpu() * std + mean
        labels_visual = labels_visual / 255.
        results_visual = results_visual / 255.
        # Save original images, labels, results
        visual = torch.cat([images, labels_visual, results_visual], dim=2)
        visual = nn.functional.interpolate(visual, scale_factor=_scale)
        num_image = 10 if num_image > 10 else num_image
        image_grid = vutils.make_grid(visual, nrow=num_image, padding=2, scale_each=True)
        vutils.save_image(image_grid, PJ(segment_directory, f"{name}.jpg"), nrow=1)


class MTAdapt(BasicAdapt):
    def __init__(self, config):
        super().__init__(config)

        self.id_to_color = torch.tensor(
            [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
             [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
             [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
             [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]], dtype=torch.float32)

    def _initial_networks(self, config):
        init = config['init']
        # Create classification networks
        if config['classifier'] == 'ResNet101':
            self.gen_a = Res101GenShare(init=init)
            self.gen_b = Res101GenShare(init=init)
            self.dis_a = ResDis(3, config['dis'], init)
            self.dis_b = ResDis(3, config['dis'], init)
            self.dis_h = ResDis(256, config['dis_h'], init)
            self.enc_bridge1 = ResNet101EncoderBridge()
            self.classifier = ResNetClassifier(config['num_class']['cls'], init)

        # Create semantic segmentation networks
        if config['segmentor'] == 'ResNet101':
            self.enc_bridge2 = ResNet101EncoderBridge()
            self.segmentor = FCN8sSegment(config['num_class']['seg'], init)

        # Load pretrained model and freeze
        if config['pretrained_checkpoint']:
            self._pretrained(config)
            self._freeze(config['freeze_list'])

        if 'enc' in config['FBN']:
            print("FBN enc")
            self._freeze_batch_norm_cls(self.gen_b.enc)
        if 'bdg2' in config['FBN']:
            print("FBN bdg2")
            self._freeze_batch_norm(self.enc_bridge2)
        if 'bdg1' in config['FBN']:
            print("FBN bdg1")
            self._freeze_batch_norm_cls(self.enc_bridge1)

    def _freeze_batch_norm(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.affine = False
                m.track_running_stats = False

    def _freeze_batch_norm_cls(self, model):
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.affine = False
                m.track_running_stats = False

    def _initial_optimizer(self, config):
        print("In DShare_2 _initial_optimizer")
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters()) + list(self.dis_h.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=config['lr'], betas=(0.5, 0.999), weight_decay=config['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=config['lr'], betas=(0.5, 0.999), weight_decay=config['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, config)
        self.gen_scheduler = get_scheduler(self.gen_opt, config)
        print(f"Initial learning rate: dis_opt: {self.dis_opt.param_groups[0]['lr']}, gen_opt: {self.gen_opt.param_groups[0]['lr']}\n")

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def classify(self, x, domain):
        """ Classification forward """
        self.eval()
        gen = self.gen_a if domain == "source" else self.gen_b
        h, _ = gen.encode(x, seg=False, s=False)
        h = self.enc_bridge1(h, seg=False)
        predicts = self.classifier(h).type(torch.float)
        return predicts

    def segment(self, x, domain):
        """ Segmetation forward """
        self.eval()
        gen = self.gen_a if domain == "source" else self.gen_b
        h, _ = gen.encode(x, seg=True, s=True)
        pool3, pool4, pool5 = self.enc_bridge2(h, seg=True)
        predicts = self.segmentor(pool3, pool4, pool5, x.shape).type(torch.float)
        predicts = torch.argmax(predicts, dim=1, keepdim=True)
        return predicts

    #################################################################
    # Training 3S2
    #################################################################
    def gen_update(self, x_a, x_b, targets, config):
        cls_targets, seg_targets = targets
        x_a1, x_a2 = x_a
        x_b1, x_b2 = x_b
        ###################################
        # Translate learning
        ###################################
        # Initialize
        x_a1_t = F.interpolate(x_a1, scale_factor=0.5)
        x_b1_t = F.interpolate(x_b1, scale_factor=0.5)
        x_a2_t = F.interpolate(x_a2, scale_factor=0.5)
        x_b2_t = F.interpolate(x_b2, scale_factor=0.5)
        t_shape_a1, t_shape_b1 = x_a1_t.shape, x_b1_t.shape
        t_shape_a2, t_shape_b2 = x_a2_t.shape, x_b2_t.shape

        self.loss_gen_total = 0
        self.gen_opt.zero_grad()

        # Encode to latent space
        h_a1, n_a1 = self.gen_a.encode(x_a1_t, seg=False, s=False)
        h_b1, n_b1 = self.gen_b.encode(x_b1_t, seg=False, s=False)
        h_a2, n_a2 = self.gen_a.encode(x_a2_t, seg=False, s=True)
        h_b2, n_b2 = self.gen_b.encode(x_b2_t, seg=False, s=True)

        # Decode to other domain (Translate)
        x_ab1 = self.gen_b.decode(h_a1, t_shape_b1, s=False)
        x_ba1 = self.gen_a.decode(h_b1, t_shape_a1, s=False)
        x_ab2 = self.gen_b.decode(h_a2, t_shape_b2, s=True)
        x_ba2 = self.gen_a.decode(h_b2, t_shape_a2, s=True)

        # [Q_cyc] Cycle reconstruction loss
        if config['cyc_w']:
            # Encode to latent again
            h_a_recon1, n_a_recon1 = self.gen_b.encode(x_ab1, seg=False, s=False)
            h_b_recon1, n_b_recon1 = self.gen_a.encode(x_ba1, seg=False, s=False)
            h_a_recon2, n_a_recon2 = self.gen_b.encode(x_ab2, seg=False, s=True)
            h_b_recon2, n_b_recon2 = self.gen_a.encode(x_ba2, seg=False, s=True)

            # Decode to original space (Cycle reconstruct)
            x_aba1 = self.gen_a.decode(h_a_recon1 + n_a_recon1, t_shape_a1, s=False)
            x_bab1 = self.gen_b.decode(h_b_recon1 + n_b_recon1, t_shape_b1, s=False)
            x_aba2 = self.gen_a.decode(h_a_recon2 + n_a_recon2, t_shape_a2, s=True)
            x_bab2 = self.gen_b.decode(h_b_recon2 + n_b_recon2, t_shape_b2, s=True)

            self.loss_gen_cyc_a = self.recon_criterion(x_aba1, x_a1_t)
            self.loss_gen_cyc_b = self.recon_criterion(x_bab1, x_b1_t)
            self.loss_gen_cyc_a += self.recon_criterion(x_aba2, x_a2_t)
            self.loss_gen_cyc_b += self.recon_criterion(x_bab2, x_b2_t)
        else:
            self.loss_gen_cyc_a, self.loss_gen_cyc_b = 0, 0

        # [Q_R] Reconstruction loss
        # Decode to image space (Reconstruct)
        x_a_recon1 = self.gen_a.decode(h_a1 + n_a1, t_shape_a1, s=False)
        x_b_recon1 = self.gen_b.decode(h_b1 + n_b1, t_shape_b1, s=False)
        x_a_recon2 = self.gen_a.decode(h_a2 + n_a2, t_shape_a2, s=True)
        x_b_recon2 = self.gen_b.decode(h_b2 + n_b2, t_shape_b2, s=True)

        self.loss_gen_recon_a = self.recon_criterion(x_a_recon1, x_a1_t)
        self.loss_gen_recon_b = self.recon_criterion(x_b_recon1, x_b1_t)
        self.loss_gen_recon_a += self.recon_criterion(x_a_recon2, x_a2_t)
        self.loss_gen_recon_b += self.recon_criterion(x_b_recon2, x_b2_t)

        # [Q_LG] Latent generator adversarial loss
        self.loss_gen_adv_h = self.dis_h.calc_gen_loss(h_b1) if config['latent_gen_w'] else 0
        self.loss_gen_adv_h += self.dis_h.calc_gen_loss(h_b2) if config['latent_gen_w'] else 0

        # [Q_TG] Translation generator loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba1)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab1)
        self.loss_gen_adv_a += self.dis_a.calc_gen_loss(x_ba2)
        self.loss_gen_adv_b += self.dis_b.calc_gen_loss(x_ab2)

        # [Q_S] Supervised loss
        self.loss_cls = 0

        # [Q_S] Supervised loss
        self.loss_seg = 0

        self.loss_gen_total =\
            config['cyc_w'] * self.loss_gen_cyc_a +\
            config['cyc_w'] * self.loss_gen_cyc_b +\
            config['recon_w'] * self.loss_gen_recon_a +\
            config['recon_w'] * self.loss_gen_recon_b +\
            config['latent_gen_w'] * self.loss_gen_adv_h +\
            config['tran_gen_w'] * self.loss_gen_adv_a +\
            config['tran_gen_w'] * self.loss_gen_adv_b +\
            config['cls_w'] * self.loss_cls +\
            config['seg_w'] * self.loss_seg

        self.loss_gen_total.backward()
        self.gen_opt.step()
        self.gen_opt.zero_grad()

    def dis_update(self, x_a, x_b, config):
        x_a1, x_a2 = x_a
        x_b1, x_b2 = x_b

        x_a1_t = F.interpolate(x_a1, scale_factor=0.5)
        x_b1_t = F.interpolate(x_b1, scale_factor=0.5)
        x_a2_t = F.interpolate(x_a2, scale_factor=0.5)
        x_b2_t = F.interpolate(x_b2, scale_factor=0.5)
        t_shape_a1, t_shape_b1 = x_a1_t.shape, x_b1_t.shape
        t_shape_a2, t_shape_b2 = x_a2_t.shape, x_b2_t.shape

        # Initialize
        self.loss_dis_total = 0
        self.dis_opt.zero_grad()

        # Encode to latent space
        h_a, n_a = self.gen_a.encode(x_a1_t, seg=False, s=False)
        h_b, n_b = self.gen_b.encode(x_b1_t, seg=False, s=False)
        h_a2, n_a2 = self.gen_a.encode(x_a2_t, seg=False, s=True)
        h_b2, n_b2 = self.gen_b.encode(x_b2_t, seg=False, s=True)

        # Decode to other domain (Translate)
        x_ab = self.gen_b.decode(h_a, t_shape_b1, s=False)
        x_ba = self.gen_a.decode(h_b, t_shape_a1, s=False)
        x_ab2 = self.gen_b.decode(h_a2, t_shape_b2, s=True)
        x_ba2 = self.gen_a.decode(h_b2, t_shape_a2, s=True)

        # [Q_TD] Translate discriminator loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a1_t)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b1_t)
        self.loss_dis_a += self.dis_a.calc_dis_loss(x_ba2.detach(), x_a2_t)
        self.loss_dis_b += self.dis_b.calc_dis_loss(x_ab2.detach(), x_b2_t)

        # [Q_LD] Latent discriminator loss
        self.loss_dis_h = self.dis_h.calc_dis_loss(h_b.detach(), h_a.detach()) if config['latent_dis_w'] else 0
        self.loss_dis_h += self.dis_h.calc_dis_loss(h_b2.detach(), h_a2.detach()) if config['latent_dis_w'] else 0

        # New loss
        self.loss_dis_tran_h = 0

        # Total loss
        self.loss_dis_total =\
            config['tran_dis_w'] * self.loss_dis_a +\
            config['tran_dis_w'] * self.loss_dis_b +\
            config['latent_dis_w'] * self.loss_dis_h +\
            config['tran_h_w'] * self.loss_dis_tran_h

        self.loss_dis_total.backward()
        self.dis_opt.step()
        self.dis_opt.zero_grad()

    ################################################################################
    # Saving and loading model
    ################################################################################
    def save(self, config, snapshot_dir, iterations):
        """ Save generators, discriminators, classifier/segmentor and optimizers """
        gen_name = PJ(snapshot_dir, f'gen_{iterations:08d}.pt')
        bdg1_name = PJ(snapshot_dir, f'bdg1_{iterations:08d}.pt')
        bdg2_name = PJ(snapshot_dir, f'bdg2_{iterations:08d}.pt')
        dis_name = PJ(snapshot_dir, f'dis_{iterations:08d}.pt')
        cls_name = PJ(snapshot_dir, f'cls_{iterations:08d}.pt')
        seg_name = PJ(snapshot_dir, f'seg_{iterations:08d}.pt')
        opt_name = PJ(snapshot_dir, 'optimizer.pt')

        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict(), 'h': self.dis_h.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)

        if config['classifier']:
            torch.save(self.classifier.state_dict(), cls_name)
        if config['segmentor']:
            torch.save(self.segmentor.state_dict(), seg_name)
        if config['classifier'] or config['segmentor']:
            torch.save(self.enc_bridge1.state_dict(), bdg1_name)
            torch.save(self.enc_bridge2.state_dict(), bdg2_name)
        print(f"Saving models in {snapshot_dir} finished.")

    def resume(self, config, checkpoint_dir):
        """ Resume training training progress. """
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])

        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        self.dis_h.load_state_dict(state_dict['h'])

        # Load Classifier
        if config['classifier']:
            last_model_name = get_model_list(checkpoint_dir, "cls")
            state_dict = torch.load(last_model_name)
            self.classifier.load_state_dict(state_dict)
        # Load Segmentor
        if config['segmentor']:
            last_model_name = get_model_list(checkpoint_dir, "seg")
            state_dict = torch.load(last_model_name)
            self.segmentor.load_state_dict(state_dict)
        # Load encoder bridge
        if config['classifier'] or config['segmentor']:
            last_model_name = get_model_list(checkpoint_dir, "bdg1")
            state_dict = torch.load(last_model_name)
            self.enc_bridge1.load_state_dict(state_dict)
            last_model_name = get_model_list(checkpoint_dir, "bdg2")
            state_dict = torch.load(last_model_name)
            self.enc_bridge2.load_state_dict(state_dict)

        # Load optimizers
        state_dict = torch.load(PJ(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, config, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, config, iterations)
        print(f'Resume from iteration {iterations}')
        return iterations

    def _pretrained(self, config):
        """ Load pretrained model. """
        print("Load pretrained model:")
        if config['gen_weights']:
            enc_name = PJ(config['pretrained_checkpoint'], config['gen_weights'])
            state_dict = torch.load(enc_name)
            self.gen_a.load_state_dict(state_dict['a'])
            self.gen_b.load_state_dict(state_dict['b'])

            if not config['enc']:
                print("(Not pretrained enc)")
                import torchvision.models as models
                model = models.resnet101(True)
                self.gen_a.enc.conv_block1 = nn.Sequential(*[model.conv1, model.bn1, model.relu, model.maxpool])
                self.gen_a.enc.conv_block2 = model.layer1
                model = models.resnet101(True)
                self.gen_a.enc.conv_block1_2 = nn.Sequential(*[model.conv1, model.bn1, model.relu, model.maxpool])

                model = models.resnet101(True)
                self.gen_b.enc.conv_block1 = nn.Sequential(*[model.conv1, model.bn1, model.relu, model.maxpool])
                self.gen_b.enc.conv_block2 = model.layer1
                model = models.resnet101(True)
                self.gen_b.enc.conv_block1_2 = nn.Sequential(*[model.conv1, model.bn1, model.relu, model.maxpool])
            print(f"{config['gen_weights']} ", end="  ")

        if config['enc_weights']:
            enc_name = PJ(config['pretrained_checkpoint'], config['enc_weights'])
            state_dict = torch.load(enc_name)
            self.gen_a.enc.load_state_dict(state_dict)
            self.gen_b.enc.load_state_dict(state_dict)
            print(f"{config['enc_weights']} ", end="  ")

        if config['dis_weights']:
            enc_name = PJ(config['pretrained_checkpoint'], config['dis_weights'])
            state_dict = torch.load(enc_name)
            self.dis_a.load_state_dict(state_dict['a'])
            self.dis_b.load_state_dict(state_dict['b'])
            self.dis_h.load_state_dict(state_dict['h'])
            print(f"{config['dis_weights']} ", end="  ")

        if config['bdg_weights']:
            bdg_name = PJ(config['pretrained_checkpoint'], config['bdg_weights'])
            state_dict = torch.load(bdg_name)
            self.enc_bridge1.load_state_dict(state_dict)
            print(f"{config['bdg_weights']} ", end="  ")

        if config['bdg2_weights']:
            bdg_name = PJ(config['pretrained_checkpoint'], config['bdg2_weights'])
            state_dict = torch.load(bdg_name)
            self.enc_bridge2.load_state_dict(state_dict)
            print(f"{config['bdg2_weights']} ", end="  ")

        if config['cls_weights']:
            cls_name = PJ(config['pretrained_checkpoint'], config['cls_weights'])
            state_dict = torch.load(cls_name)
            self.classifier.load_state_dict(state_dict)
            print(f"{config['cls_weights']} ", end="  ")

        if config['seg_weights']:
            seg_name = PJ(config['pretrained_checkpoint'], config['seg_weights'])
            state_dict = torch.load(seg_name)
            self.segmentor.load_state_dict(state_dict)
            print(f"{config['seg_weights']} ", end="  ")
        print(f"\nLoad pretrained model from {config['pretrained_checkpoint']}")

    #################################################################
    # Record images
    #################################################################
    def sample(self, x_a, x_b, task):
        self.eval()
        torch.set_grad_enabled(False)

        n = min(x_a.size(0), 5)
        x_a, x_b = x_a[:n], x_b[:n]

        t_shape_a, t_shape_b = x_a.shape, x_b.shape
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(n):
            if task == 'cls':
                h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0), seg=False, s=False)
                h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0), seg=False, s=False)
                x_a_recon.append(self.gen_a.decode(h_a, t_shape_a, s=False))
                x_b_recon.append(self.gen_b.decode(h_b, t_shape_b, s=False))
                x_ba.append(self.gen_a.decode(h_b, t_shape_a, s=False))
                x_ab.append(self.gen_b.decode(h_a, t_shape_b, s=False))
            else:
                h_a, _ = self.gen_a.encode(x_a[i].unsqueeze(0), seg=False, s=True)
                h_b, _ = self.gen_b.encode(x_b[i].unsqueeze(0), seg=False, s=True)
                x_a_recon.append(self.gen_a.decode(h_a, t_shape_a, s=True))
                x_b_recon.append(self.gen_b.decode(h_b, t_shape_b, s=True))
                x_ba.append(self.gen_a.decode(h_b, t_shape_a, s=True))
                x_ab.append(self.gen_b.decode(h_a, t_shape_b, s=True))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba, x_ab = torch.cat(x_ba), torch.cat(x_ab)
        self.train()
        torch.set_grad_enabled(True)
        return x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba

    def record_translate_images(self, x_a, x_b, image_directory, name, _scale=0.5):
        mean = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
        x_a, x_a_recon, x_ab, x_b, x_b_recon, x_ba = self.sample(x_a, x_b, task=name[-3:])
        num_image, _, h, w = x_a.shape

        # Denormalize
        visual_a = torch.cat([x_a, x_a_recon, x_ab], dim=0).cpu()
        visual_b = torch.cat([x_b, x_b_recon, x_ba], dim=0).cpu()
        visual_a = visual_a * std + mean
        visual_b = visual_b * std + mean
        # Resize
        visual_a = nn.functional.interpolate(visual_a, scale_factor=_scale)
        visual_b = nn.functional.interpolate(visual_b, scale_factor=_scale)
        # Combine
        visual_a = vutils.make_grid(visual_a, nrow=num_image, padding=2, scale_each=True)
        visual_b = vutils.make_grid(visual_b, nrow=num_image, padding=2, scale_each=True)
        # Save results
        vutils.save_image(visual_a, PJ(image_directory, f"{name}_a.jpg"), nrow=1)
        vutils.save_image(visual_b, PJ(image_directory, f"{name}_b.jpg"), nrow=1)
