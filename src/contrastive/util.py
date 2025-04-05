import math
import torch
from torchvision import transforms
import random
import operator
import numpy as np
import os
import wandb
import torch.nn.functional as F

from models.estimators import AgeEstimator
from pathlib import Path


class NViewTransform:
    """Create N augmented views of the same image"""

    def __init__(self, transform, N):
        self.transform = transform
        self.N = N

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.N)]


class Crop(object):
    """ Crop the given n-dimensional array either at a random location or
    centered.
    """

    def __init__(self, shape, type="center", keep_dim=False):
        assert type in ["center", "random"]
        self.shape = shape
        self.cropping_type = type
        self.keep_dim = keep_dim

    def slow_crop(self, X):
        img_shape = np.array(X.shape)

        if type(self.shape) == int:
            size = [self.shape for _ in range(len(self.shape))]
        else:
            size = np.copy(self.shape)

        # print('img_shape:', img_shape, 'size', size)

        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]

            if self.cropping_type == "center":
                delta_before = int((img_shape[ndim] - size[ndim]) / 2.0)

            elif self.cropping_type == "random":
                delta_before = np.random.randint(0, img_shape[ndim] - size[ndim] + 1)

            indexes.append(slice(delta_before, delta_before + size[ndim]))

        if self.keep_dim:
            mask = np.zeros(img_shape, dtype=np.bool)
            mask[tuple(indexes)] = True
            arr_copy = X.copy()
            arr_copy[~mask] = 0
            return arr_copy

        _X = X[tuple(indexes)]
        # print('cropped.shape', _X.shape)
        return _X

    def fast_crop(self, X):
        # X is a single image (CxWxHxZ)
        shape = X.shape

        delta = [shape[1] - self.shape[1],
                 shape[2] - self.shape[2],
                 shape[3] - self.shape[3]]

        if self.cropping_type == "center":
            offset = list(map(operator.floordiv, delta, [2] * len(delta)))
            X = X[:, offset[0]:offset[0] + self.shape[1],
                offset[1]:offset[1] + self.shape[2],
                offset[2]:offset[2] + self.shape[3]]

        elif self.cropping_type == "random":
            offset = [
                int(random.random() * 128) % (delta[0] + 1),
                int(random.random() * 128) % (delta[1] + 1),
                int(random.random() * 128) % (delta[2] + 1)
            ]
            X = X[:, offset[0]:offset[0] + self.shape[1],
                offset[1]:offset[1] + self.shape[2],
                offset[2]:offset[2] + self.shape[3]]
        else:
            raise ValueError("Invalid cropping_type", self.cropping_type)

        return X

    def __call__(self, X):
        return self.fast_crop(X)


class Cutout(object):
    """Apply a cutout on the images
    cf. Improved Regularization of Convolutional Neural Networks with Cutout, arXiv, 2017
    We assume that the square to be cut is inside the image.
    """

    def __init__(self, patch_size=None, value=0, random_size=False, inplace=False, localization=None, probability=0.5):
        self.patch_size = patch_size
        self.value = value
        self.random_size = random_size
        self.inplace = inplace
        self.localization = localization
        self.probability = probability

    def __call__(self, arr):
        if np.random.rand() >= self.probability:
            return arr

        img_shape = np.array(arr.shape)
        if type(self.patch_size) == int:
            size = [self.patch_size for _ in range(len(img_shape))]
        else:
            size = np.copy(self.patch_size)
        assert len(size) == len(img_shape), "Incorrect patch dimension."
        indexes = []
        for ndim in range(len(img_shape)):
            if size[ndim] > img_shape[ndim] or size[ndim] < 0:
                size[ndim] = img_shape[ndim]
            if self.random_size:
                size[ndim] = np.random.randint(0, size[ndim])
            if self.localization is not None:
                delta_before = max(self.localization[ndim] - size[ndim] // 2, 0)
            else:
                delta_before = np.random.randint(0, img_shape[ndim] - size[ndim] + 1)
            indexes.append(slice(int(delta_before), int(delta_before + size[ndim])))
        if self.inplace:
            arr[tuple(indexes)] = self.value
            return arr
        else:
            arr_cut = np.copy(arr)
            arr_cut[tuple(indexes)] = self.value
            return arr_cut


class Pad(object):
    """ Pad the given n-dimensional array
    """

    def __init__(self, shape, **kwargs):
        self.shape = shape
        self.kwargs = kwargs

    def __call__(self, X):
        _X = self._apply_padding(X)
        return _X

    def _apply_padding(self, arr):
        orig_shape = arr.shape
        padding = []
        for orig_i, final_i in zip(orig_shape, self.shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0:
                padding.append([half_shape_i, half_shape_i])
            else:
                padding.append([half_shape_i, half_shape_i + 1])
        for cnt in range(len(arr.shape) - len(padding)):
            padding.append([0, 0])
        fill_arr = np.pad(arr, padding, **self.kwargs)
        return fill_arr


class AddGaussianNoise(object):
    """Add Gaussian noise to a NumPy array."""

    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, np_array):
        noise = np.random.normal(self.mean, self.std, np_array.shape)
        np_array_noisy = np_array + noise
        return np_array_noisy


class AddGaussianNoiseTensor(object):
    """Add Gaussian noise to a PyTorch tensor, intended for use on the GPU."""

    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Ensure the noise is created on the same device as the input tensor
        device = tensor.device
        noise = torch.randn(tensor.size(), device=device) * self.std + self.mean
        tensor_noisy = tensor + noise
        return tensor_noisy


def get_transforms(opts):
    to_tensor = transforms.Lambda(lambda x: torch.tensor(x).float())
    unsqueeze_np = transforms.Lambda(lambda x: np.expand_dims(x, axis=0))
    squeeze = transforms.Lambda(lambda x: x.squeeze(0))

    if opts.tf == 'none':
        aug = transforms.Compose([
            transforms.Lambda(lambda x: x),
            to_tensor,
        ])

    elif opts.tf == 'crop':
        aug = transforms.Compose([
            unsqueeze_np,
            Crop((1, 91, 91, 91), type="random"),
            Pad((1, 109, 91, 109)),
            to_tensor,
            squeeze
        ])

    elif opts.tf == 'cutout':
        aug = transforms.Compose([
            unsqueeze_np,
            Cutout(patch_size=[1, 32, 32, 32], probability=0.5),
            to_tensor,
            squeeze
        ])

    elif opts.tf == 'noise':
        aug = transforms.Compose([
            to_tensor,
            AddGaussianNoiseTensor(mean=0., std=opts.noise_std),
        ])

    elif opts.tf == 'all':
        aug = transforms.Compose([
            unsqueeze_np,
            Cutout(patch_size=[1, 32, 32, 32], probability=0.5),
            Crop((1, 91, 91, 91), type="random"),
            Pad((1, 109, 91, 109)),
            to_tensor,
            squeeze
        ])

    T_train = transforms.Compose([
        aug
    ])

    T_test = transforms.Compose([
        to_tensor
    ])

    return T_train, T_test


def arg2bool(val):
    if isinstance(val, bool):
        return val

    elif isinstance(val, str):
        if val == "true":
            return True

        if val == "false":
            return False

    val = int(val)
    assert val == 0 or val == 1
    return val == 1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


class MAE():
    def __init__(self):
        self.reset()

    def reset(self):
        self.outputs = []
        self.targets = []
        self.avg = np.inf

    def update(self, outputs, targets):
        self.outputs.append(outputs.detach())
        self.targets.append(targets.detach())
        self.avg = F.l1_loss(torch.cat(self.outputs, 0), torch.cat(self.targets, 0))


class Accuracy():
    def __init__(self, topk=(1,)):
        self.reset()
        self.topk = topk

    def reset(self):
        self.outputs = []
        self.targets = []
        self.avg = np.inf

    def update(self, outputs, targets):
        self.outputs.append(outputs.detach())
        self.targets.append(targets.detach())
        self.avg = accuracy(torch.cat(self.outputs, 0), torch.cat(self.targets, 0), self.topk)


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=True)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state_dict = model.state_dict()
    if torch.cuda.device_count() > 1:
        state_dict = model.module.state_dict()

    state = {
        'opts': opt,
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'run_id': wandb.run.id
    }
    torch.save(state, save_file)
    del state


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.lr_decay == 'cosine':
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


@torch.no_grad()
def gather_age_feats(model, dataloader, opts):
    features = []
    age_labels = []

    model.eval()
    for idx, (images, labels, _) in enumerate(dataloader):
        if isinstance(images, list):
            images = images[0]
        images = images.to(opts.device)
        images = torch.unsqueeze(images, 1)
        features.append(model.features(images))
        age_labels.append(labels)

    return torch.cat(features, 0).cpu().numpy(), torch.cat(age_labels, 0).cpu().numpy()


@torch.no_grad()
def gather_age_feats_sites(model, dataloader, opts):
    features = []
    age_labels = []

    sex_l = []
    sites_l = []
    imbalances_l = []
    coverages_l = []

    model.eval()
    for idx, (images, labels, (sex, sites, imbalances, MRE_coverages)) in enumerate(dataloader):
        if isinstance(images, list):
            images = images[0]
        images = images.to(opts.device)
        images = torch.unsqueeze(images, 1)

        features.append(model.features(images))
        age_labels.append(labels)

        sex_l.append(list(sex))
        sites_l.append(list(sites))
        imbalances_l.append(list(imbalances))
        coverages_l.append(list(MRE_coverages))

    features = torch.cat(features, 0).cpu().numpy()
    age_labels = torch.cat(age_labels, 0).cpu().numpy()

    sex_l = np.array([item for sublist in sex_l for item in sublist])
    sites_l = np.array([item for sublist in sites_l for item in sublist])
    imbalances_l = np.array([item for sublist in imbalances_l for item in sublist])
    coverages_l = np.array([item for sublist in coverages_l for item in sublist])

    return features, age_labels, sex_l, sites_l, imbalances_l, coverages_l


@torch.no_grad()
def compute_age_mae(model, train_loader, test, opts):
    site_estimator = AgeEstimator()

    print("Training age estimator")
    train_X, train_y = gather_age_feats(model, train_loader, opts)
    mae_train = site_estimator.fit(train_X, train_y)

    print("Computing BA")
    test_X, test_y = gather_age_feats(model, test, opts)
    mae_test = site_estimator.score(test_X, test_y)

    return mae_train, mae_test


@torch.no_grad()
def gather_site_feats(model, dataloader, opts):
    features = []
    site_labels = []

    model.eval()
    for idx, (images, _, sites) in enumerate(dataloader):
        if isinstance(images, list):
            images = images[0]
        images = images.to(opts.device)
        features.append(model.features(images))
        site_labels.append(sites)

    return torch.cat(features, 0).cpu().numpy(), torch.cat(site_labels, 0).cpu().numpy()


def calculate_dynamic_NN_nb(current_epoch, step_size, end_NN_nb, max_epochs=50, start_NN_nb=32):
    """
    Calculate the dynamic NN_nb for the current epoch

    Args:
    - current_epoch (int): The current epoch number.

    Returns:
    - int: The dynamically calculated NN_nb for the current epoch.
    """
    # Calculate the number of steps completed
    steps_completed = current_epoch // step_size

    # Total number of steps in the training
    total_steps = max_epochs // step_size

    # Calculate decrease per step
    NN_nb_decrement_per_step = (start_NN_nb - end_NN_nb) / float(total_steps - 1)

    # Calculate current NN_nb based on the steps completed
    dynamic_NN_nb = int(start_NN_nb - (NN_nb_decrement_per_step * steps_completed))

    # Ensure dynamic_NN_nb does not go below end_NN_nb
    dynamic_NN_nb = max(dynamic_NN_nb, end_NN_nb)

    return dynamic_NN_nb


@torch.no_grad()
def gather_age_feats(model, dataloader, opts):
    features = []
    age_labels = []

    model.eval()
    for idx, (images, labels, _) in enumerate(dataloader):

        if opts.modality == 'dr+stiffness':
            images_dr, images_stiffness = images

            if isinstance(images_dr, list):
                images_dr = images_dr[0]
                images_stiffness = images_stiffness[0]

            images_dr = torch.unsqueeze(images_dr, 1).to(opts.device)
            images_stiffness = torch.unsqueeze(images_stiffness, 1).to(opts.device)
            images = torch.cat([images_dr, images_stiffness], dim=1)

        else:
            if isinstance(images, list):
                images = images[0]
            images = images.to(opts.device)
            images = torch.unsqueeze(images, 1)

        if opts.features_w_projection:
            features.append(model.features(images))
        else:
            print("features_wo_projection")
            features.append(model.features_wo_projection(images))

        age_labels.append(labels)

    return torch.cat(features, 0).cpu().numpy(), torch.cat(age_labels, 0).cpu().numpy()


@torch.no_grad()
def gather_age_ID_feats(model, dataloader, opts):
    features = []
    age_labels = []
    IDs_l = []

    model.eval()
    for idx, (images, labels, IDs) in enumerate(dataloader):

        if opts.modality == 'dr+stiffness':
            images_dr, images_stiffness = images

            if isinstance(images_dr, list):
                images_dr = images_dr[0]
                images_stiffness = images_stiffness[0]

            images_dr = torch.unsqueeze(images_dr, 1).to(opts.device)
            images_stiffness = torch.unsqueeze(images_stiffness, 1).to(opts.device)
            images = torch.cat([images_dr, images_stiffness], dim=1)

        else:
            if isinstance(images, list):
                images = images[0]
            images = images.to(opts.device)
            images = torch.unsqueeze(images, 1)

        if opts.features_w_projection:
            features.append(model.features(images))
        else:
            features.append(model.features_wo_projection(images))

        age_labels.append(labels)
        IDs_l.append(IDs)

    return torch.cat(features, 0).cpu().numpy(), torch.cat(age_labels, 0).cpu().numpy(), np.array(IDs_l)