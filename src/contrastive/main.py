import datetime
import math
import os
import torch
import torch.utils.data
import argparse
import models
import losses
import time
import wandb
import json
import numpy as np

from torch import nn
from util import AverageMeter, NViewTransform, ensure_dir, set_seed, arg2bool, save_model, get_transforms
from util import warmup_learning_rate, adjust_learning_rate
from util import compute_age_mae, gather_age_feats, gather_age_ID_feats
from data import MREDataset, MREDatasetStudy, MREDatasetRegion, MREDatasetStudyRegion


def parse_arguments():
    parser = argparse.ArgumentParser(description="Weakly contrastive learning for brain age predictin",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Misc
    parser.add_argument('--device', type=str, help='torch device', default='cuda')
    parser.add_argument('--print_freq', type=int, help='print frequency', default=10)
    parser.add_argument('--trial', type=int, help='random seed / trial id', default=0)
    parser.add_argument('--fold', type=int, help='fold (of 5 CV)', default=0)
    parser.add_argument('--save_dir', type=str, help='output dir', default='output')
    parser.add_argument('--save_freq', type=int, help='save frequency', default=50)
    parser.add_argument('--data_dir', type=str, help='path of data dir', default='/data')
    parser.add_argument('--amp', type=arg2bool, help='use amp', default=False)
    parser.add_argument('--clip_grad', type=arg2bool, help='clip gradient to prevent nan', default=False)
    parser.add_argument('--wandb_name', type=str, help='wandb project name', default='contrastive-brain-age-prediction')
    parser.add_argument('--save_model', type=arg2bool, help='save model', default=False)

    # Model
    parser.add_argument('--model', type=str, help='model architecture', default='resnet18')

    # Optimizer
    parser.add_argument('--epochs', type=int, help='number of epochs', default=300)
    parser.add_argument('--batch_size', type=int, help='batch size', default=256)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
    parser.add_argument('--lr_decay', type=str, help='type of decay', choices=['cosine', 'step'], default='step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='decay rate for learning rate (for step)')
    parser.add_argument('--lr_decay_epochs', type=str, help='steps of lr decay (list)', default="700,800,900")
    parser.add_argument('--lr_decay_step', type=int, help='decay rate step (overwrites lr_decay_epochs', default=10)
    parser.add_argument('--warm', type=arg2bool, help='warmup lr', default=False)
    parser.add_argument('--optimizer', type=str, help="optimizer (adam or sgd)", choices=["adam", "sgd"],
                        default="adam")
    parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=5e-5)

    # for ablations
    parser.add_argument('--features_w_projection', type=arg2bool, help='not use projection head to get features',
                        default=True)
    parser.add_argument('--distance_weight_param', type=float, help='distance weight parameter', default=math.inf)

    # Data
    parser.add_argument('--tf', type=str, help='data augmentation',
                        choices=['none', 'crop', 'cutout', 'all', 'noise', 'rotation', 'flip', 'shift', 'scale'],
                        default='none')
    parser.add_argument('--noise_std', type=float, help='std for noise augmentation', default=0.05)
    parser.add_argument('--modality', type=str, help='imaging modality',
                        choices=['T1', 'dr', 'stiffness', 'dr+stiffness'], default='stiffness')
    parser.add_argument('--study', type=str, help='study', default='ATLAS+NITRC1+NITRC2+OA+MIMS+BMI+NOVA')
    parser.add_argument('--segmentation', type=str, help='brain segmentation: WM/GM', default='None')
    parser.add_argument('--brain_region', type=str, help='subcortical brain structures', default='None')
    parser.add_argument('--norm_level', type=str, help='normalization level', default='dataset_level')

    # Loss
    parser.add_argument('--method', type=str, help='loss function',
                        choices=['supcon', 'yaware', 'threshold', 'expw', 'SupNNCLR'], default='supcon')
    parser.add_argument('--kernel', type=str, help='Kernel function (not for supcon)',
                        choices=['cauchy', 'gaussian', 'rbf'], default=None)
    parser.add_argument('--delta_reduction', type=str,
                        help='use mean or sum to reduce 3d delta mask (only for method=threshold)', default='sum')
    parser.add_argument('--temp', type=float, help='loss temperature', default=0.1)
    parser.add_argument('--alpha', type=float, help='infonce weight', default=1.)
    parser.add_argument('--sigma', type=float, help='gaussian-rbf kernel sigma / cauchy gamma', default=1)
    parser.add_argument('--n_views', type=int, help='num. of multiviews', default=2)

    # hyperparameter for new modifications
    parser.add_argument('--NN_nb_step_size', type=int, help='step size for NN_nb', default=0)
    parser.add_argument('--end_NN_nb', type=int, help='label type', default=4)
    parser.add_argument('--NN_nb_selection', type=str, help='selection method for NN_nb',
                        choices=['euclidean', 'similarity', 'manhattan', 'chebyshev', 'no'], default='similarity')

    opts = parser.parse_args()

    if opts.batch_size > 256:
        print("Forcing warm")
        opts.warm = True

    if opts.lr_decay_step is not None:
        opts.lr_decay_epochs = list(range(opts.lr_decay_step, opts.epochs, opts.lr_decay_step))
        print(f"Computed decay epochs based on step ({opts.lr_decay_step}):", opts.lr_decay_epochs)
    else:
        iterations = opts.lr_decay_epochs.split(',')
        opts.lr_decay_epochs = list([])
        for it in iterations:
            opts.lr_decay_epochs.append(int(it))

    if opts.warm:
        opts.warmup_from = 0.01
        opts.warm_epochs = 10
        if opts.lr_decay == 'cosine':
            eta_min = opts.lr * (opts.lr_decay_rate ** 3)
            opts.warmup_to = eta_min + (opts.lr - eta_min) * (
                    1 + math.cos(math.pi * opts.warm_epochs / opts.epochs)) / 2
        else:
            opts.milestones = [int(s) for s in opts.lr_decay_epochs.split(',')]
            opts.warmup_to = opts.lr

    if opts.method == 'supcon':
        print('method == supcon, binning age')
        opts.label = 'bin'
    else:
        print('method != supcon, using real age value')
        opts.label = 'cont'

    if opts.method == 'supcon' and opts.kernel is not None:
        print('Invalid kernel for supcon')
        exit(0)

    if opts.method != 'supcon' and opts.kernel is None:
        print('Kernel cannot be None for method != supcon')
        exit(1)

    if opts.model == 'densenet121':
        opts.n_views = 1

    if torch.cuda.is_available():
        print('on GPU')
        device = 'cuda'
        opts.device = device
        opts.location = 'cluster'
    else:
        print('not on GPU')
        device = 'cpu'
        opts.device = device
        opts.location = 'local'

    return opts


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_data(opts):
    T_train, T_test = get_transforms(opts)
    T_train = NViewTransform(T_train, opts.n_views)

    if opts.brain_region == 'None':
        train_dataset = MREDataset(modality=opts.modality, segmentation=opts.segmentation, transform=T_train,
                                   train=True,
                                   location=opts.location, fold=opts.fold)
    else:
        train_dataset = MREDatasetRegion(modality=opts.modality, brain_region=opts.brain_region, transform=T_train,
                                         train=True, location=opts.location, fold=opts.fold)

    if opts.norm_level == 'image_level':
        train_dataset.norm_on_image_level()
    elif opts.norm_level == 'dataset_level':
        train_dataset.norm()
    else:
        raise ValueError("Norm level not recognized")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)

    if opts.brain_region == 'None':
        train_dataset_score = MREDataset(modality=opts.modality, segmentation=opts.segmentation, transform=T_train,
                                         train=True, location=opts.location, fold=opts.fold)
    else:
        train_dataset_score = MREDatasetRegion(modality=opts.modality, brain_region=opts.brain_region,
                                               transform=T_train,
                                               train=True, location=opts.location, fold=opts.fold)

    if opts.norm_level == 'image_level':
        train_dataset_score.norm_on_image_level()
    else:
        train_dataset_score.norm()

    train_loader_score = torch.utils.data.DataLoader(train_dataset_score, batch_size=opts.batch_size, shuffle=False)

    if opts.brain_region == 'None':
        test_dataset = MREDataset(modality=opts.modality, segmentation=opts.segmentation, transform=T_test, train=False,
                                  location=opts.location, fold=opts.fold)
    else:
        test_dataset = MREDatasetRegion(modality=opts.modality, brain_region=opts.brain_region, transform=T_test,
                                        train=False, location=opts.location, fold=opts.fold)

    if opts.norm_level == 'image_level':
        test_dataset.norm_on_image_level()
    else:
        test_dataset.norm()

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False)

    return train_loader, train_loader_score, test_loader


def load_data_cohorts(opts):
    if opts.study == 'ATLAS+NITRC1+NITRC2+MIMS+BMI+NOVA':
        study_healthy = 'OA'
        study_disease = 'AD'

    elif opts.study == 'ATLAS+NITRC1+NITRC2+OA+BMI+NOVA':
        study_healthy = 'MIMS'
        study_disease = 'MCI'

    else:
        raise ValueError("Study not recognized")

    T_train, T_test = get_transforms(opts)
    T_train = NViewTransform(T_train, opts.n_views)

    if opts.brain_region == 'None':
        train_dataset = MREDataset(modality=opts.modality, segmentation=opts.segmentation, transform=T_train,
                                   train=True,
                                   location=opts.location, fold=opts.fold, study=opts.study)
    else:
        train_dataset = MREDatasetRegion(modality=opts.modality, brain_region=opts.brain_region, transform=T_train,
                                         train=True, location=opts.location, fold=opts.fold, study=opts.study)

    if opts.norm_level == 'image_level':
        train_dataset.norm_on_image_level()
    else:
        train_dataset.norm()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)

    if opts.brain_region == 'None':
        train_dataset_score = MREDataset(modality=opts.modality, segmentation=opts.segmentation, transform=T_train,
                                         train=True, location=opts.location, fold=opts.fold, study=opts.study)
    else:
        train_dataset_score = MREDatasetRegion(modality=opts.modality, brain_region=opts.brain_region,
                                               transform=T_train,
                                               train=True, location=opts.location, fold=opts.fold, study=opts.study)

    if opts.norm_level == 'image_level':
        train_dataset_score.norm_on_image_level()
    else:
        train_dataset_score.norm()

    train_loader_score = torch.utils.data.DataLoader(train_dataset_score, batch_size=opts.batch_size, shuffle=False)

    if opts.brain_region == 'None':
        test_dataset = MREDataset(modality=opts.modality, segmentation=opts.segmentation, transform=T_test, train=False,
                                  location=opts.location, fold=opts.fold, study=opts.study)
    else:
        test_dataset = MREDatasetRegion(modality=opts.modality, brain_region=opts.brain_region, transform=T_test,
                                        train=False, location=opts.location, fold=opts.fold, study=opts.study)

    if opts.norm_level == 'image_level':
        test_dataset.norm_on_image_level()
    else:
        test_dataset.norm()

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opts.batch_size, shuffle=False)

    if opts.modality == 'dr+stiffness':
        mu_stiff = train_dataset.mu_stiff
        sigma_stiff = train_dataset.sigma_stiff

        mu_dr = train_dataset.mu_dr
        sigma_dr = train_dataset.sigma_dr

    elif opts.modality == 'stiffness':
        mu_stiff = train_dataset.mu
        sigma_stiff = train_dataset.sigma

        mu_dr = None
        sigma_dr = None

    elif opts.modality == 'dr':
        mu_dr = train_dataset.mu
        sigma_dr = train_dataset.sigma

        mu_stiff = None
        sigma_stiff = None

    elif opts.modality == 'T1':
        mu_stiff = None
        sigma_stiff = None

        mu_dr = None
        sigma_dr = None

    if opts.brain_region == 'None':
        healthy_dataset = MREDatasetStudy(modality=opts.modality, study=study_healthy, transform=T_test,
                                          location=opts.location)
    else:
        healthy_dataset = MREDatasetStudyRegion(modality=opts.modality, brain_region=opts.brain_region,
                                                transform=T_test,
                                                location=opts.location, study=study_healthy)

    healthy_dataset.norm(mu_stiff=mu_stiff, sigma_stiff=sigma_stiff, mu_dr=mu_dr, sigma_dr=sigma_dr)

    healthy_loader = torch.utils.data.DataLoader(healthy_dataset, batch_size=opts.batch_size, shuffle=False)

    if opts.brain_region == 'None':
        disease_dataset = MREDatasetStudy(modality=opts.modality, study=study_disease, transform=T_test,
                                          location=opts.location)
    else:
        disease_dataset = MREDatasetStudyRegion(modality=opts.modality, brain_region=opts.brain_region,
                                                transform=T_test,
                                                location=opts.location, study=study_disease)

    disease_dataset.norm(mu_stiff=mu_stiff, sigma_stiff=sigma_stiff, mu_dr=mu_dr, sigma_dr=sigma_dr)

    disease_loader = torch.utils.data.DataLoader(disease_dataset, batch_size=opts.batch_size, shuffle=False)

    return train_loader, train_loader_score, test_loader, healthy_loader, disease_loader


def load_model(opts):
    if opts.modality == 'dr+stiffness':
        opts.model = 'resnet18_inChannels2'

    if 'resnet' in opts.model:
        model = models.SupConResNet(opts.model, feat_dim=128)

    if opts.device == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using multiple CUDA devices ({torch.cuda.device_count()})")
        model = torch.nn.DataParallel(model)

    model = model.to(opts.device)

    def gaussian_kernel(x):
        x = x - x.T
        return torch.exp(-(x ** 2) / (2 * (opts.sigma ** 2))) / (math.sqrt(2 * torch.pi) * opts.sigma)

    def rbf(x):
        x = x - x.T
        return torch.exp(-(x ** 2) / (2 * (opts.sigma ** 2)))

    def cauchy(x):
        x = x - x.T
        return 1. / (opts.sigma * (x ** 2) + 1)

    kernels = {
        'none': None,
        'cauchy': cauchy,
        'gaussian': gaussian_kernel,
        'rbf': rbf
    }

    infonce = losses.DynLocRep_loss(method=opts.method, temperature=opts.temp, kernel=kernels[opts.kernel],
                                    delta_reduction=opts.delta_reduction, epochs=opts.epochs,
                                    NN_nb_step_size=opts.NN_nb_step_size, end_NN_nb=opts.end_NN_nb,
                                    NN_nb_selection=opts.NN_nb_selection)

    infonce = infonce.to(opts.device)

    return model, infonce


def load_optimizer(model, opts):
    if opts.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr,
                                    momentum=opts.momentum,
                                    weight_decay=opts.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

    return optimizer


def train(train_loader, model, infonce, optimizer, opts, epoch):
    loss = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    scaler = torch.cuda.amp.GradScaler() if opts.amp else None
    model.train()

    t1 = time.time()
    for idx, (images, labels, _) in enumerate(
            train_loader):  # for idx, (images, labels, (sex, site, imbalances)) in enumerate(train_loader):

        data_time.update(time.time() - t1)

        if opts.modality == 'dr+stiffness':

            images_dr, images_stiffness = images

            images_dr = torch.cat(images_dr, dim=0).to(opts.device)
            images_dr = torch.unsqueeze(images_dr, 1)

            images_stiffness = torch.cat(images_stiffness, dim=0).to(opts.device)
            images_stiffness = torch.unsqueeze(images_stiffness, 1)

            images = torch.cat([images_dr, images_stiffness], dim=1)

            bsz = labels.shape[0]

        else:

            images = torch.cat(images, dim=0).to(opts.device)
            images = torch.unsqueeze(images, 1)
            bsz = labels.shape[0]

        warmup_learning_rate(opts, epoch, idx, len(train_loader), optimizer)

        with torch.cuda.amp.autocast(scaler is not None):
            projected = model(images)

            projected = torch.split(projected, [bsz] * opts.n_views, dim=0)
            projected = torch.cat([f.unsqueeze(1) for f in projected], dim=1)

            if opts.NN_nb_step_size > 0:
                running_loss = infonce(features=projected,
                                       labels=labels.to(opts.device),
                                       epoch=epoch)

            else:
                running_loss = infonce(features=projected,
                                       labels=labels.to(opts.device))

        optimizer.zero_grad()
        if scaler is None:
            running_loss.backward()
            if opts.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
        else:
            scaler.scale(running_loss).backward()
            if opts.clip_grad:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            scaler.step(optimizer)
            scaler.update()

        loss.update(running_loss.item(), bsz)
        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(train_loader) - idx)

        if (idx + 1) % opts.print_freq == 0:
            print(f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]:\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"loss {loss.avg:.3f}\t")

    return loss.avg, batch_time.avg, data_time.avg


if __name__ == '__main__':
    opts = parse_arguments()

    set_seed(opts.trial)

    if opts.study == 'ATLAS+NITRC1+NITRC2+OA+MIMS+BMI+NOVA':
        train_loader, train_loader_score, test_loader = load_data(opts)
    else:
        train_loader, train_loader_score, test_loader, healthy_loader, disease_loader = load_data_cohorts(opts)

    model, infonce = load_model(opts)
    optimizer = load_optimizer(model, opts)

    model_name = opts.model
    if opts.warm:
        model_name = f"{model_name}_warm"
    if opts.amp:
        model_name = f"{model_name}_amp"

    method_name = opts.method
    if opts.method == 'threshold':
        method_name = f"{method_name}_reduction_{opts.delta_reduction}"

    optimizer_name = opts.optimizer
    if opts.clip_grad:
        optimizer_name = f"{optimizer_name}_clipgrad"

    kernel_name = opts.kernel
    if opts.kernel == "gaussian" or opts.kernel == 'rbf':
        kernel_name = f"{kernel_name}_sigma{opts.sigma}"
    elif opts.kernel == 'cauchy':
        kernel_name = f"{kernel_name}_gamma{opts.sigma}"

    prefix = 'from_scratch'

    run_name = (f"_contrastive_{prefix}_{opts.modality}_{opts.norm_level}_Segm_{opts.segmentation}_Region_"
                f"{opts.brain_region}_{opts.study}_{method_name}_E{opts.epochs}_trial{opts.trial}_fold{opts.fold}"
                f"_NN_nb_selection_{opts.NN_nb_selection}_end_NN_nb_{opts.end_NN_nb}_NN_nb_step_size_{opts.NN_nb_step_size}")


    save_dir = os.path.join(opts.save_dir, f"models", run_name)

    if opts.save_model:
        ensure_dir(save_dir)

    opts.model_class = model.__class__.__name__
    opts.criterion = infonce.__class__.__name__
    opts.optimizer_class = optimizer.__class__.__name__

    wandb.init(entity='jakobwandb', project=opts.wandb_name, config=opts, name=run_name,  # sync_tensorboard=True,
               settings=wandb.Settings(code_dir="/src"), tags=['to test'])

    if opts.amp:
        print("Using AMP")

    start_time = time.time()
    best_acc = 0.
    for epoch in range(1, opts.epochs + 1):
        adjust_learning_rate(opts, optimizer, epoch)

        t1 = time.time()
        loss_train, batch_time, data_time = train(train_loader, model, infonce, optimizer, opts, epoch)
        t2 = time.time()
        wandb.log({"train/loss": loss_train, "lr": optimizer.param_groups[0]['lr'], "BT": batch_time, "DT": data_time,
                   "epoch": epoch})

        print(f"epoch {epoch}, total time {t2 - start_time:.2f}, epoch time {t2 - t1:.3f} loss {loss_train:.4f}")

        if opts.study == 'ATLAS+NITRC1+NITRC2+OA+MIMS+BMI+NOVA':
            mae_train, mae_test = compute_age_mae(model, train_loader_score, test_loader, opts)
            wandb.log({"train/mae": mae_train, "test/mae": mae_test, "epoch": epoch})
        else:
            mae_train, mae_test = compute_age_mae(model, train_loader_score, test_loader, opts)
            _, mae_healthy = compute_age_mae(model, train_loader_score, healthy_loader, opts)
            _, mae_disease = compute_age_mae(model, train_loader_score, disease_loader, opts)
            wandb.log(
                {"train/mae": mae_train, "test/mae": mae_test, "healthy/mae": mae_healthy, "disease/mae": mae_disease,
                 "epoch": epoch})

        if epoch % opts.save_freq == 0:
            if opts.save_model:
                print(f"SAVING MODEL... (mae_test: {mae_test:.2f})")
                save_file = os.path.join(save_dir, f"ckpt_epoch_{epoch}.pth")
                save_model(model, optimizer, opts, epoch, save_file)

            if opts.study == 'ATLAS+NITRC1+NITRC2+OA+MIMS+BMI+NOVA':
                print("Age MAE:", mae_train, mae_test)
            else:
                print("Age MAE:", mae_train, mae_test, mae_healthy, mae_disease)

    if opts.study == 'ATLAS+NITRC1+NITRC2+OA+MIMS+BMI+NOVA':
        mae_train, mae_test = compute_age_mae(model, train_loader_score, test_loader, opts)
        wandb.log({"train/mae": mae_train, "test/mae": mae_test, "epoch": epoch})

    else:
        age_estimator = models.AgeEstimator()

        train_X, train_y, train_ID = gather_age_ID_feats(model, train_loader_score, opts)
        mae_train = age_estimator.fit(train_X, train_y)
        train_y_train = age_estimator.predict(train_X)

        test_X, test_y, test_ID = gather_age_ID_feats(model, test_loader, opts)
        test_y_pred = age_estimator.predict(test_X)
        mae_test = age_estimator.score(test_X, test_y)

        healthy_X, healthy_y, healthy_ID = gather_age_ID_feats(model, healthy_loader, opts)
        healthy_y_pred = age_estimator.predict(healthy_X)
        mae_healthy = age_estimator.score(healthy_X, healthy_y)

        disease_X, disease_y, disease_ID = gather_age_ID_feats(model, disease_loader, opts)
        disease_y_pred = age_estimator.predict(disease_X)
        mae_disease = age_estimator.score(disease_X, disease_y)

        # create directory
        pred_dir = 'placeholder_dir' #set pred_dir to the desired directory
        ensure_dir(os.path.join(pred_dir, run_name))

        # save predictions
        np.save(os.path.join(pred_dir, run_name, 'train_y.npy'), train_y)
        np.save(os.path.join(pred_dir, run_name, 'train_y_pred.npy'), train_y_train)
        np.save(os.path.join(pred_dir, run_name, 'train_ID.npy'), train_ID)

        np.save(os.path.join(pred_dir, run_name, 'test_y.npy'), test_y)
        np.save(os.path.join(pred_dir, run_name, 'test_y_pred.npy'), test_y_pred)
        np.save(os.path.join(pred_dir, run_name, 'test_ID.npy'), test_ID)

        np.save(os.path.join(pred_dir, run_name, 'healthy_y.npy'), healthy_y)
        np.save(os.path.join(pred_dir, run_name, 'healthy_y_pred.npy'), healthy_y_pred)
        np.save(os.path.join(pred_dir, run_name, 'healthy_ID.npy'), healthy_ID)

        np.save(os.path.join(pred_dir, run_name, 'disease_y.npy'), disease_y)
        np.save(os.path.join(pred_dir, run_name, 'disease_y_pred.npy'), disease_y_pred)
        np.save(os.path.join(pred_dir, run_name, 'disease_ID.npy'), disease_ID)

        wandb.log({"train/mae": mae_train, "test/mae": mae_test, "healthy/mae": mae_healthy, "disease/mae": mae_disease,
                   "epoch": epoch})