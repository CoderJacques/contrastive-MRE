import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import os
import nibabel as nib
import scipy
import re


class MREDataset(torch.utils.data.Dataset):
    def __init__(self, modality, segmentation=None, transform=None, train=True, location='local', fold=0,
                 study='ATLAS+NITRC1+NITRC2+OA+MIMS+BMI+NOVA'):

        if segmentation in ['GM', 'WM']:
            (stiffness, dr, T1, age, sex, study, id) = load_segmentations(study=study, segmentation=segmentation,
                                                                          location=location)

        else:
            (stiffness, dr, T1, age, sex, study, id) = load_samples(study=study, location=location)

        if modality == 'stiffness':
            _, mu_stiff, sigma_stiff = normalize_mean_0_std_1(stiffness, default_value=0, mu_nonzero=None,
                                                              sigma_nonzero=None)
            [self.mu, self.sigma] = mu_stiff, sigma_stiff

        elif modality == 'dr':
            _, mu_dr, sigma_dr = normalize_mean_0_std_1(dr, default_value=0, mu_nonzero=None, sigma_nonzero=None)
            [self.mu, self.sigma] = mu_dr, sigma_dr

        elif modality == 'dr+stiffness':
            _, mu_stiff, sigma_stiff = normalize_mean_0_std_1(stiffness, default_value=0, mu_nonzero=None,
                                                              sigma_nonzero=None)
            _, mu_dr, sigma_dr = normalize_mean_0_std_1(dr, default_value=0, mu_nonzero=None, sigma_nonzero=None)

            [self.mu_stiff, self.sigma_stiff] = mu_stiff, sigma_stiff
            [self.mu_dr, self.sigma_dr] = mu_dr, sigma_dr

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        assert fold in range(5) or fold is None

        if fold in range(5):
            for fold_iter, (train_ids, test_ids) in enumerate(kfold.split(stiffness)):

                if fold_iter == fold:
                    stiffness_train, stiffness_test = stiffness[train_ids], stiffness[test_ids]
                    dr_train, dr_test = dr[train_ids], dr[test_ids]
                    T1_train, T1_test = T1[train_ids], T1[test_ids]
                    sex_train, sex_test = sex[train_ids], sex[test_ids]
                    age_train, age_test = age[train_ids], age[test_ids]
                    study_train, study_test = study[train_ids], study[test_ids]

                else:
                    continue
        else:
            stiffness_train, stiffness_test, \
                dr_train, dr_test, \
                T1_train, T1_test, \
                age_train, age_test, \
                sex_train, sex_test, \
                study_train, study_test = train_test_split(stiffness, dr, T1, age, sex, study, test_size=0.2,
                                                           random_state=42)

        if train:
            self.y = age_train
            self.sex = sex_train
            self.site = study_train

            if modality == 'stiffness':
                self.x = stiffness_train
            elif modality == 'dr':
                self.x = dr_train
            elif modality == 'T1':
                self.x = T1_train
            elif modality == 'dr+stiffness':
                self.x_dr = dr_train
                self.x_stiff = stiffness_train

        else:
            self.y = age_test
            self.sex = sex_test
            self.site = study_test

            if modality == 'stiffness':
                self.x = stiffness_test
            elif modality == 'dr':
                self.x = dr_test
            elif modality == 'T1':
                self.x = T1_test
            elif modality == 'dr+stiffness':
                self.x_dr = dr_test
                self.x_stiff = stiffness_test

        self.modality = modality
        self.T = transform

    def norm(self):

        default_value = 0

        if self.modality == 'T1':
            self.x = norm_whole_batch(self.x, 'mean_std', default_value)

        elif self.modality == 'dr' or self.modality == 'stiffness':
            self.x, _, _ = normalize_mean_0_std_1(self.x,
                                                  default_value=default_value,
                                                  mu_nonzero=self.mu,
                                                  sigma_nonzero=self.sigma)
        elif self.modality == 'dr+stiffness':
            self.x_dr, _, _ = normalize_mean_0_std_1(self.x_dr,
                                                     default_value=default_value,
                                                     mu_nonzero=self.mu_dr,
                                                     sigma_nonzero=self.sigma_dr)
            self.x_stiff, _, _ = normalize_mean_0_std_1(self.x_stiff,
                                                        default_value=default_value,
                                                        mu_nonzero=self.mu_stiff,
                                                        sigma_nonzero=self.sigma_stiff)

        else:
            raise ValueError('Invalid modality')

    def norm_on_image_level(self):

        default_value = 0

        if self.modality in ['T1', 'dr', 'stiffness']:
            self.x = norm_whole_batch(self.x, 'mean_std', default_value)

        elif self.modality == 'dr+stiffness':
            self.x_dr = norm_whole_batch(self.x_dr, 'mean_std', default_value)
            self.x_stiff = norm_whole_batch(self.x_stiff, 'mean_std', default_value)

        else:
            raise ValueError('Invalid modality')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        y = self.y[index]

        sex = self.sex[index]
        site = self.site[index]

        if self.modality == 'dr+stiffness':

            x_dr = self.x_dr[index]
            x_stiff = self.x_stiff[index]

            if self.T is not None:
                x_dr = self.T(x_dr)
                x_stiff = self.T(x_stiff)
            else:
                x_dr = torch.from_numpy(x_dr).float()
                x_stiff = torch.from_numpy(x_stiff).float()

            return (x_dr, x_stiff), y, (sex, site)

        else:
            x = self.x[index]

            if self.T is not None:
                x = self.T(x)
            else:
                x = torch.from_numpy(x).float()

            return x, y, (sex, site)


class MREDatasetRegion(torch.utils.data.Dataset):
    def __init__(self, modality, brain_region=None, transform=None, train=True, location='local', fold=0,
                 study='ATLAS+NITRC1+NITRC2+OA+MIMS+BMI+NOVA'):

        (stiffness, dr, age, ID) = load_region(study=study, location=location, region=brain_region)

        if modality == 'stiffness':
            _, mu_stiff, sigma_stiff = normalize_mean_0_std_1(stiffness, default_value=0, mu_nonzero=None,
                                                              sigma_nonzero=None)
            [self.mu, self.sigma] = mu_stiff, sigma_stiff

        elif modality == 'dr':
            _, mu_dr, sigma_dr = normalize_mean_0_std_1(dr, default_value=0, mu_nonzero=None, sigma_nonzero=None)
            [self.mu, self.sigma] = mu_dr, sigma_dr

        elif modality == 'dr+stiffness':
            _, mu_stiff, sigma_stiff = normalize_mean_0_std_1(stiffness, default_value=0, mu_nonzero=None,
                                                              sigma_nonzero=None)
            _, mu_dr, sigma_dr = normalize_mean_0_std_1(dr, default_value=0, mu_nonzero=None, sigma_nonzero=None)

            [self.mu_stiff, self.sigma_stiff] = mu_stiff, sigma_stiff
            [self.mu_dr, self.sigma_dr] = mu_dr, sigma_dr

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        assert fold in range(5) or fold is None

        if fold in range(5):
            for fold_iter, (train_ids, test_ids) in enumerate(kfold.split(stiffness)):

                if fold_iter == fold:
                    stiffness_train, stiffness_test = stiffness[train_ids], stiffness[test_ids]
                    dr_train, dr_test = dr[train_ids], dr[test_ids]
                    age_train, age_test = age[train_ids], age[test_ids]
                    ID_train, ID_test = ID[train_ids], ID[test_ids]

                else:
                    continue
        else:
            stiffness_train, stiffness_test, \
                dr_train, dr_test, \
                age_train, age_test, \
                ID_train, ID_test = train_test_split(stiffness, dr, age, ID, test_size=0.2, random_state=42)

        if train:
            self.y = age_train
            self.ID = ID_train

            if modality == 'stiffness':
                self.x = stiffness_train
            elif modality == 'dr':
                self.x = dr_train
            elif modality == 'dr+stiffness':
                self.x_dr = dr_train
                self.x_stiff = stiffness_train

        else:
            self.y = age_test
            self.ID = ID_test

            if modality == 'stiffness':
                self.x = stiffness_test
            elif modality == 'dr':
                self.x = dr_test
            elif modality == 'dr+stiffness':
                self.x_dr = dr_test
                self.x_stiff = stiffness_test

        self.modality = modality
        self.T = transform

    def norm(self):

        default_value = 0

        if self.modality == 'dr' or self.modality == 'stiffness':
            self.x, _, _ = normalize_mean_0_std_1(self.x,
                                                  default_value=default_value,
                                                  mu_nonzero=self.mu,
                                                  sigma_nonzero=self.sigma)
        elif self.modality == 'dr+stiffness':
            self.x_dr, _, _ = normalize_mean_0_std_1(self.x_dr,
                                                     default_value=default_value,
                                                     mu_nonzero=self.mu_dr,
                                                     sigma_nonzero=self.sigma_dr)
            self.x_stiff, _, _ = normalize_mean_0_std_1(self.x_stiff,
                                                        default_value=default_value,
                                                        mu_nonzero=self.mu_stiff,
                                                        sigma_nonzero=self.sigma_stiff)

        else:
            raise ValueError('Invalid modality')

    def norm_on_image_level(self):

        default_value = 0

        if self.modality in ['T1', 'dr', 'stiffness']:
            self.x = norm_whole_batch(self.x, 'mean_std', default_value)

        elif self.modality == 'dr+stiffness':
            self.x_dr = norm_whole_batch(self.x_dr, 'mean_std', default_value)
            self.x_stiff = norm_whole_batch(self.x_stiff, 'mean_std', default_value)

        else:
            raise ValueError('Invalid modality')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        y = self.y[index]
        ID = self.ID[index]

        if self.modality == 'dr+stiffness':

            x_dr = self.x_dr[index]
            x_stiff = self.x_stiff[index]

            if self.T is not None:
                x_dr = self.T(x_dr)
                x_stiff = self.T(x_stiff)
            else:
                x_dr = torch.from_numpy(x_dr).float()
                x_stiff = torch.from_numpy(x_stiff).float()

            return (x_dr, x_stiff), y, ID

        else:
            x = self.x[index]

            if self.T is not None:
                x = self.T(x)
            else:
                x = torch.from_numpy(x).float()

            return x, y, ID


class MREDatasetStudy(torch.utils.data.Dataset):
    def __init__(self, modality, study, transform=None, location='local'):

        (stiffness, dr, T1, age, sex, study, id) = load_samples(study=study, location=location)

        self.y = age
        self.sex = sex
        self.site = study

        if modality == 'stiffness':
            self.x = stiffness
        elif modality == 'dr':
            self.x = dr
        elif modality == 'T1':
            self.x = T1
        elif modality == 'dr+stiffness':
            self.x_dr = dr
            self.x_stiff = stiffness

        self.modality = modality
        self.T = transform

    def norm(self, mu_stiff=None, sigma_stiff=None, mu_dr=None, sigma_dr=None):

        default_value = 0

        if self.modality == 'T1':
            self.x = norm_whole_batch(self.x, 'mean_std', default_value)

        elif self.modality == 'dr' or self.modality == 'stiffness':

            mu = mu_stiff if self.modality == 'stiffness' else mu_dr
            sigma = sigma_stiff if self.modality == 'stiffness' else sigma_dr

            self.x, _, _ = normalize_mean_0_std_1(self.x,
                                                  default_value=default_value,
                                                  mu_nonzero=mu,
                                                  sigma_nonzero=sigma)

        elif self.modality == 'dr+stiffness':
            self.x_dr, _, _ = normalize_mean_0_std_1(self.x_dr,
                                                     default_value=default_value,
                                                     mu_nonzero=mu_dr,
                                                     sigma_nonzero=sigma_dr)
            self.x_stiff, _, _ = normalize_mean_0_std_1(self.x_stiff,
                                                        default_value=default_value,
                                                        mu_nonzero=mu_stiff,
                                                        sigma_nonzero=sigma_stiff)

        else:
            raise ValueError('Invalid modality')

    def norm_on_image_level(self):

        default_value = 0

        if self.modality in ['T1', 'dr', 'stiffness']:
            self.x = norm_whole_batch(self.x, 'mean_std', default_value)

        elif self.modality == 'dr+stiffness':
            self.x_dr = norm_whole_batch(self.x_dr, 'mean_std', default_value)
            self.x_stiff = norm_whole_batch(self.x_stiff, 'mean_std', default_value)

        else:
            raise ValueError('Invalid modality')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        y = self.y[index]

        sex = self.sex[index]
        site = self.site[index]

        if self.modality == 'dr+stiffness':

            x_dr = self.x_dr[index]
            x_stiff = self.x_stiff[index]

            if self.T is not None:
                x_dr = self.T(x_dr)
                x_stiff = self.T(x_stiff)
            else:
                x_dr = torch.from_numpy(x_dr).float()
                x_stiff = torch.from_numpy(x_stiff).float()

            return (x_dr, x_stiff), y, (sex, site)

        else:
            x = self.x[index]

            if self.T is not None:
                x = self.T(x)
            else:
                x = torch.from_numpy(x).float()

            return x, y, (sex, site)


class MREDatasetStudyRegion(torch.utils.data.Dataset):
    def __init__(self, modality, study, brain_region, transform=None, location='local'):

        (stiffness, dr, age, ID) = load_region(study=study, location=location, region=brain_region)

        self.y = age
        self.ID = ID

        if modality == 'stiffness':
            self.x = stiffness
        elif modality == 'dr':
            self.x = dr
        elif modality == 'dr+stiffness':
            self.x_dr = dr
            self.x_stiff = stiffness

        self.modality = modality
        self.T = transform

    def norm(self, mu_stiff=None, sigma_stiff=None, mu_dr=None, sigma_dr=None):
        print('norm')

        default_value = 0

        if self.modality == 'dr' or self.modality == 'stiffness':

            mu = mu_stiff if self.modality == 'stiffness' else mu_dr
            sigma = sigma_stiff if self.modality == 'stiffness' else sigma_dr

            self.x, _, _ = normalize_mean_0_std_1(self.x,
                                                  default_value=default_value,
                                                  mu_nonzero=mu,
                                                  sigma_nonzero=sigma)

        elif self.modality == 'dr+stiffness':
            self.x_dr, _, _ = normalize_mean_0_std_1(self.x_dr,
                                                     default_value=default_value,
                                                     mu_nonzero=mu_dr,
                                                     sigma_nonzero=sigma_dr)
            self.x_stiff, _, _ = normalize_mean_0_std_1(self.x_stiff,
                                                        default_value=default_value,
                                                        mu_nonzero=mu_stiff,
                                                        sigma_nonzero=sigma_stiff)

        else:
            raise ValueError('Invalid modality')

    def norm_on_image_level(self):
        print('norm_on_image_level')

        default_value = 0

        if self.modality in ['T1', 'dr', 'stiffness']:
            self.x = norm_whole_batch(self.x, 'mean_std', default_value)

        elif self.modality == 'dr+stiffness':
            self.x_dr = norm_whole_batch(self.x_dr, 'mean_std', default_value)
            self.x_stiff = norm_whole_batch(self.x_stiff, 'mean_std', default_value)

        else:
            raise ValueError('Invalid modality')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        y = self.y[index]
        ID = self.ID[index]

        if self.modality == 'dr+stiffness':

            x_dr = self.x_dr[index]
            x_stiff = self.x_stiff[index]

            if self.T is not None:
                x_dr = self.T(x_dr)
                x_stiff = self.T(x_stiff)
            else:
                x_dr = torch.from_numpy(x_dr).float()
                x_stiff = torch.from_numpy(x_stiff).float()

            return (x_dr, x_stiff), y

        else:
            x = self.x[index]

            if self.T is not None:
                x = self.T(x)
            else:
                x = torch.from_numpy(x).float()

            return x, y, ID


def norm_whole_batch(batch, norm, default_value):
    batch_normed = np.zeros_like(batch)

    for i in range(batch.shape[0]):
        if norm == 'mean_std':
            batch_normed[i], _, _ = normalize_mean_0_std_1(batch[i], default_value, None, None)

        else:
            raise ValueError('norm has to be min_max or mean_std')

    return batch_normed


def normalize_mean_0_std_1(arr, default_value, mu_nonzero, sigma_nonzero):
    arr_nonzero = arr[np.nonzero(arr)]

    if mu_nonzero is None and sigma_nonzero is None:
        mu_nonzero = np.mean(arr_nonzero)
        sigma_nonzero = np.std(arr_nonzero)

    if default_value == 0:
        arr_pp = np.zeros_like(arr)

    elif default_value == -1:
        arr_pp = np.ones_like(arr) * -1

    else:
        raise ValueError('default_value has to be 0 or -1')

    arr_pp[np.nonzero(arr)] = (arr[np.nonzero(arr)] - mu_nonzero) / sigma_nonzero

    return arr_pp, mu_nonzero, sigma_nonzero


def load_samples(study, location):
    if location == 'local':
        prefix_path = '/Users/jakobtraeuble/PycharmProjects/BrainAgeMRE/Data'

    elif location == 'cluster':
        prefix_path = '/home/jnt27/rds/hpc-work/MRE'

    else:
        raise ValueError('specific location as local or cluster')

    stiffness_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/stiffness_134.npy', allow_pickle=True)
    dr_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/dr_134.npy', allow_pickle=True)
    T1_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/T1_masked_ATLAS.npy', allow_pickle=True)  # T1_ATLAS.npy
    age_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/age_ATLAS.npy', allow_pickle=True)
    sex_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/sex_ATLAS.npy', allow_pickle=True)
    id_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/id_ATLAS.npy', allow_pickle=True)
    study_ATLAS = np.array(['ATLAS'] * len(age_ATLAS))

    stiffness_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/stiffness_OA.npy', allow_pickle=True)
    dr_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/dr_OA.npy', allow_pickle=True)
    T1_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/T1_masked_OA.npy', allow_pickle=True)  # T1_OA.npy
    age_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/age_OA.npy', allow_pickle=True)
    sex_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/sex_OA.npy', allow_pickle=True)
    id_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/id_OA.npy', allow_pickle=True)
    study_OA = np.array(['CN'] * len(age_OA))

    stiffness_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/stiffness_AD.npy', allow_pickle=True)
    dr_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/dr_AD.npy', allow_pickle=True)
    T1_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/T1_masked_AD.npy', allow_pickle=True)
    age_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/age_AD.npy', allow_pickle=True)
    sex_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/sex_AD.npy', allow_pickle=True)
    id_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/id_AD.npy', allow_pickle=True)
    study_AD = np.array(['AD'] * len(age_OA))

    stiffness_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/stiffness_BMI.npy', allow_pickle=True)
    dr_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/dr_BMI.npy', allow_pickle=True)
    T1_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/T1_masked_BMI.npy', allow_pickle=True)  # T1_MIMS.npy
    age_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/age_BMI.npy', allow_pickle=True)
    sex_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/sex_BMI.npy', allow_pickle=True)
    id_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/id_BMI.npy', allow_pickle=True)
    study_BMI = np.array(['BMI'] * len(age_BMI))

    stiffness_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/stiffness_NOVA.npy', allow_pickle=True)
    dr_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/dr_NOVA.npy', allow_pickle=True)
    T1_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/T1_masked_NOVA.npy', allow_pickle=True)  # T1_MIMS.npy
    age_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/age_NOVA.npy', allow_pickle=True)
    sex_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/sex_NOVA.npy', allow_pickle=True)
    id_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/id_NOVA.npy', allow_pickle=True)
    study_NOVA = np.array(['NOVA'] * len(age_NOVA))

    stiffness_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/stiffness_NITRC_batch_1.npy',
                                      allow_pickle=True)
    dr_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/dr_NITRC_batch_1.npy', allow_pickle=True)
    T1_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/T1_masked_NITRC_batch_1.npy',
                               allow_pickle=True)  # T1_NITRC_batch_1.npy
    age_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/age_NITRC_batch_1.npy', allow_pickle=True)
    sex_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/sex_NITRC_batch_1.npy', allow_pickle=True)
    id_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/id_NITRC_batch_1.npy', allow_pickle=True)
    study_NITRC_batch_1 = np.array(['NITRC_batch_1'] * len(age_NITRC_batch_1))

    stiffness_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/stiffness_NITRC_batch_2.npy',
                                      allow_pickle=True)
    dr_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/dr_NITRC_batch_2.npy', allow_pickle=True)
    T1_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/T1_masked_NITRC_batch_2.npy',
                               allow_pickle=True)  # T1_NITRC_batch_2.npy
    age_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/age_NITRC_batch_2.npy', allow_pickle=True)
    sex_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/sex_NITRC_batch_2.npy', allow_pickle=True)
    id_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/id_NITRC_batch_2.npy', allow_pickle=True)
    study_NITRC_batch_2 = np.array(['NITRC_batch_2'] * len(age_NITRC_batch_2))

    stiffness_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/stiffness_MIMS.npy', allow_pickle=True)
    dr_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/dr_MIMS.npy', allow_pickle=True)
    T1_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/T1_masked_MIMS.npy', allow_pickle=True)  # T1_MIMS.npy
    age_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/age_MIMS.npy', allow_pickle=True)
    sex_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/sex_MIMS.npy', allow_pickle=True)
    id_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/id_MIMS.npy', allow_pickle=True)
    study_MIMS = np.array(['MIMS'] * len(age_MIMS))

    stiffness_MIMS_MCI = np.load(prefix_path + '/Curtis_Johnson/MIMS_MCI/stiffness_MIMS_MCI.npy', allow_pickle=True)
    dr_MIMS_MCI = np.load(prefix_path + '/Curtis_Johnson/MIMS_MCI/dr_MIMS_MCI.npy', allow_pickle=True)
    T1_MIMS_MCI = np.load(prefix_path + '/Curtis_Johnson/MIMS_MCI/T1_masked_MIMS_MCI.npy', allow_pickle=True)
    age_MIMS_MCI = np.load(prefix_path + '/Curtis_Johnson/MIMS_MCI/age_MIMS_MCI.npy', allow_pickle=True)
    sex_MIMS_MCI = np.load(prefix_path + '/Curtis_Johnson/MIMS_MCI/sex_MIMS_MCI.npy', allow_pickle=True)
    id_MIMS_MCI = np.load(prefix_path + '/Curtis_Johnson/MIMS_MCI/id_MIMS_MCI.npy', allow_pickle=True)
    study_MIMS_MCI = np.array(['MIMS_MCI'] * len(age_MIMS))

    if study == 'ATLAS+NITRC1+NITRC2+OA+MIMS+BMI+NOVA':
        stiffness_all_healthy = np.concatenate(
            (stiffness_ATLAS, stiffness_NITRC_batch_1, stiffness_NITRC_batch_2, stiffness_OA,
             stiffness_MIMS, stiffness_BMI, stiffness_NOVA), axis=0)
        dr_all_healthy = np.concatenate((dr_ATLAS, dr_NITRC_batch_1, dr_NITRC_batch_2, dr_OA, dr_MIMS, dr_BMI, dr_NOVA),
                                        axis=0)
        T1_all_healthy = np.concatenate((T1_ATLAS, T1_NITRC_batch_1, T1_NITRC_batch_2, T1_OA, T1_MIMS, T1_BMI, T1_NOVA),
                                        axis=0)
        age_all_healthy = np.concatenate(
            (age_ATLAS, age_NITRC_batch_1, age_NITRC_batch_2, age_OA, age_MIMS, age_BMI, age_NOVA), axis=0)
        sex_all_healthy = np.concatenate(
            (sex_ATLAS, sex_NITRC_batch_1, sex_NITRC_batch_2, sex_OA, sex_MIMS, sex_BMI, sex_NOVA), axis=0)
        study_all_healthy = np.concatenate(
            (study_ATLAS, study_NITRC_batch_1, study_NITRC_batch_2, study_OA, study_MIMS, study_BMI, study_NOVA),
            axis=0)
        id_all_healthy = np.concatenate((id_ATLAS, id_NITRC_batch_1, id_NITRC_batch_2, id_OA, id_MIMS, id_BMI, id_NOVA),
                                        axis=0)

        unique, inverse = np.unique(age_all_healthy, return_inverse=True)
        counts = np.bincount(inverse)
        total_count = age_all_healthy.shape[0]
        imbalance_percentages = counts[inverse] / total_count

        return (
        stiffness_all_healthy, dr_all_healthy, T1_all_healthy, age_all_healthy, sex_all_healthy, study_all_healthy,
        id_all_healthy)

    elif study == 'ATLAS+NITRC1+NITRC2+MIMS+BMI+NOVA':
        stiffness = np.concatenate((stiffness_ATLAS, stiffness_NITRC_batch_1, stiffness_NITRC_batch_2, stiffness_MIMS,
                                    stiffness_BMI, stiffness_NOVA), axis=0)
        dr = np.concatenate((dr_ATLAS, dr_NITRC_batch_1, dr_NITRC_batch_2, dr_MIMS, dr_BMI, dr_NOVA), axis=0)
        T1 = np.concatenate((T1_ATLAS, T1_NITRC_batch_1, T1_NITRC_batch_2, T1_MIMS, T1_BMI, T1_NOVA), axis=0)
        age = np.concatenate((age_ATLAS, age_NITRC_batch_1, age_NITRC_batch_2, age_MIMS, age_BMI, age_NOVA), axis=0)
        sex = np.concatenate((sex_ATLAS, sex_NITRC_batch_1, sex_NITRC_batch_2, sex_MIMS, sex_BMI, sex_NOVA), axis=0)
        study = np.concatenate(
            (study_ATLAS, study_NITRC_batch_1, study_NITRC_batch_2, study_MIMS, study_BMI, study_NOVA), axis=0)
        id = np.concatenate((id_ATLAS, id_NITRC_batch_1, id_NITRC_batch_2, id_MIMS, id_BMI, id_NOVA), axis=0)

        return (stiffness, dr, T1, age, sex, study, id)

    elif study == 'AD':
        stiffness = stiffness_AD
        dr = dr_AD
        T1 = T1_AD
        age = age_AD
        sex = sex_AD
        study = study_AD
        id = id_AD

        return (stiffness, dr, T1, age, sex, study, id)

    elif study == 'OA':
        stiffness = stiffness_OA
        dr = dr_OA
        T1 = T1_OA
        age = age_OA
        sex = sex_OA
        study = study_OA
        id = id_OA

        return (stiffness, dr, T1, age, sex, study, id)

    elif study == 'ATLAS+NITRC1+NITRC2+OA+BMI+NOVA':
        stiffness = np.concatenate((stiffness_ATLAS, stiffness_NITRC_batch_1, stiffness_NITRC_batch_2, stiffness_OA,
                                    stiffness_BMI, stiffness_NOVA), axis=0)
        dr = np.concatenate((dr_ATLAS, dr_NITRC_batch_1, dr_NITRC_batch_2, dr_OA, dr_BMI, dr_NOVA), axis=0)
        T1 = np.concatenate((T1_ATLAS, T1_NITRC_batch_1, T1_NITRC_batch_2, T1_OA, T1_BMI, T1_NOVA), axis=0)
        age = np.concatenate((age_ATLAS, age_NITRC_batch_1, age_NITRC_batch_2, age_OA, age_BMI, age_NOVA), axis=0)
        sex = np.concatenate((sex_ATLAS, sex_NITRC_batch_1, sex_NITRC_batch_2, sex_OA, sex_BMI, sex_NOVA), axis=0)
        study = np.concatenate((study_ATLAS, study_NITRC_batch_1, study_NITRC_batch_2, study_OA, study_BMI, study_NOVA),
                               axis=0)
        id = np.concatenate((id_ATLAS, id_NITRC_batch_1, id_NITRC_batch_2, id_OA, id_BMI, id_NOVA), axis=0)

        return (stiffness, dr, T1, age, sex, study, id)

    elif study == 'MCI':
        stiffness = stiffness_MIMS_MCI
        dr = dr_MIMS_MCI
        T1 = T1_MIMS_MCI
        age = age_MIMS_MCI
        sex = sex_MIMS_MCI
        study = study_MIMS_MCI
        id = id_MIMS_MCI

        return (stiffness, dr, T1, age, sex, study, id)

    elif study == 'MIMS':
        stiffness = stiffness_MIMS
        dr = dr_MIMS
        T1 = T1_MIMS
        age = age_MIMS
        sex = sex_MIMS
        study = study_MIMS
        id = id_MIMS

        return (stiffness, dr, T1, age, sex, study, id)


def load_segmentations(study, segmentation, location):
    if location == 'local':
        prefix_path = '/Users/jakobtraeuble/PycharmProjects/BrainAgeMRE/Data'

    elif location == 'cluster':
        prefix_path = '/home/jnt27/rds/hpc-work/MRE'

    else:
        raise ValueError('specific location as local or cluster')

    if segmentation == 'GM':
        name_segm = '1'
    elif segmentation == 'WM':
        name_segm = '2'
    else:
        raise ValueError('Invalid segmentation')

    stiffness_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/stiffness_MNI_' + name_segm + '_ATLAS.npy',
                              allow_pickle=True)
    dr_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/DR_MNI_' + name_segm + '_ATLAS.npy', allow_pickle=True)
    T1_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/T1_MNI_' + name_segm + '_ATLAS.npy', allow_pickle=True)
    age_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/age_ATLAS.npy', allow_pickle=True)
    sex_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/sex_ATLAS.npy', allow_pickle=True)
    id_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/id_ATLAS.npy', allow_pickle=True)
    study_ATLAS = np.array(['ATLAS'] * len(age_ATLAS))

    stiffness_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/stiffness_MNI_' + name_segm + '_OA.npy', allow_pickle=True)
    dr_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/DR_MNI_' + name_segm + '_OA.npy', allow_pickle=True)
    T1_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/T1_MNI_' + name_segm + '_OA.npy', allow_pickle=True)
    age_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/age_OA.npy', allow_pickle=True)
    sex_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/sex_OA.npy', allow_pickle=True)
    id_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/id_OA.npy', allow_pickle=True)
    study_OA = np.array(['CN'] * len(age_OA))

    stiffness_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/stiffness_MNI_' + name_segm + '_AD.npy', allow_pickle=True)
    dr_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/DR_MNI_' + name_segm + '_AD.npy', allow_pickle=True)
    T1_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/T1_MNI_' + name_segm + '_AD.npy', allow_pickle=True)
    age_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/age_AD.npy', allow_pickle=True)
    sex_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/sex_AD.npy', allow_pickle=True)
    id_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/id_AD.npy', allow_pickle=True)
    study_AD = np.array(['AD'] * len(age_OA))

    stiffness_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/stiffness_MNI_' + name_segm + '_BMI.npy',
                            allow_pickle=True)
    dr_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/DR_MNI_' + name_segm + '_BMI.npy', allow_pickle=True)
    T1_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/T1_MNI_' + name_segm + '_BMI.npy', allow_pickle=True)
    age_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/age_BMI.npy', allow_pickle=True)
    sex_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/sex_BMI.npy', allow_pickle=True)
    id_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/id_BMI.npy', allow_pickle=True)
    study_BMI = np.array(['BMI'] * len(age_BMI))

    stiffness_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/stiffness_MNI_' + name_segm + '_NOVA.npy',
                             allow_pickle=True)
    dr_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/DR_MNI_' + name_segm + '_NOVA.npy', allow_pickle=True)
    T1_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/T1_MNI_' + name_segm + '_NOVA.npy', allow_pickle=True)
    age_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/age_NOVA.npy', allow_pickle=True)
    sex_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/sex_NOVA.npy', allow_pickle=True)
    id_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/id_NOVA.npy', allow_pickle=True)
    study_NOVA = np.array(['NOVA'] * len(age_NOVA))

    stiffness_NITRC_batch_1 = np.load(
        prefix_path + '/Curtis_Johnson/NITRC_batch_1/stiffness_MNI_' + name_segm + '_NITRC_batch_1.npy',
        allow_pickle=True)
    dr_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/DR_MNI_' + name_segm + '_NITRC_batch_1.npy',
                               allow_pickle=True)
    T1_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/T1_MNI_' + name_segm + '_NITRC_batch_1.npy',
                               allow_pickle=True)
    age_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/age_NITRC_batch_1.npy', allow_pickle=True)
    sex_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/sex_NITRC_batch_1.npy', allow_pickle=True)
    id_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/id_NITRC_batch_1.npy', allow_pickle=True)
    study_NITRC_batch_1 = np.array(['NITRC_batch_1'] * len(age_NITRC_batch_1))

    stiffness_NITRC_batch_2 = np.load(
        prefix_path + '/Curtis_Johnson/NITRC_batch_2/stiffness_MNI_' + name_segm + '_NITRC_batch_2.npy',
        allow_pickle=True)
    dr_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/DR_MNI_' + name_segm + '_NITRC_batch_2.npy',
                               allow_pickle=True)
    T1_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/T1_MNI_' + name_segm + '_NITRC_batch_2.npy',
                               allow_pickle=True)
    age_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/age_NITRC_batch_2.npy', allow_pickle=True)
    sex_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/sex_NITRC_batch_2.npy', allow_pickle=True)
    id_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/id_NITRC_batch_2.npy', allow_pickle=True)
    study_NITRC_batch_2 = np.array(['NITRC_batch_2'] * len(age_NITRC_batch_2))

    stiffness_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/stiffness_MNI_' + name_segm + '_MIMS.npy',
                             allow_pickle=True)
    dr_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/DR_MNI_' + name_segm + '_MIMS.npy', allow_pickle=True)
    T1_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/T1_MNI_' + name_segm + '_MIMS.npy', allow_pickle=True)
    age_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/age_MIMS.npy', allow_pickle=True)
    sex_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/sex_MIMS.npy', allow_pickle=True)
    id_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/id_MIMS.npy', allow_pickle=True)
    study_MIMS = np.array(['MIMS'] * len(age_MIMS))

    stiffness_MIMS_MCI = np.load(prefix_path + '/Curtis_Johnson/MIMS_MCI/stiffness_MNI_' + name_segm + '_MIMS_MCI.npy',
                                 allow_pickle=True)
    dr_MIMS_MCI = np.load(prefix_path + '/Curtis_Johnson/MIMS_MCI/DR_MNI_' + name_segm + '_MIMS_MCI.npy',
                          allow_pickle=True)
    T1_MIMS_MCI = np.load(prefix_path + '/Curtis_Johnson/MIMS_MCI/T1_MNI_' + name_segm + '_MIMS_MCI.npy',
                          allow_pickle=True)
    age_MIMS_MCI = np.load(prefix_path + '/Curtis_Johnson/MIMS_MCI/age_MIMS_MCI.npy', allow_pickle=True)
    sex_MIMS_MCI = np.load(prefix_path + '/Curtis_Johnson/MIMS_MCI/sex_MIMS_MCI.npy', allow_pickle=True)
    id_MIMS_MCI = np.load(prefix_path + '/Curtis_Johnson/MIMS_MCI/id_MIMS_MCI.npy', allow_pickle=True)
    study_MIMS_MCI = np.array(['MIMS_MCI'] * len(age_MIMS))

    if study == 'ATLAS+NITRC1+NITRC2+OA+MIMS+BMI+NOVA':
        stiffness_all_healthy = np.concatenate(
            (stiffness_ATLAS, stiffness_NITRC_batch_1, stiffness_NITRC_batch_2, stiffness_OA,
             stiffness_MIMS, stiffness_BMI, stiffness_NOVA), axis=0)
        dr_all_healthy = np.concatenate((dr_ATLAS, dr_NITRC_batch_1, dr_NITRC_batch_2, dr_OA, dr_MIMS, dr_BMI, dr_NOVA),
                                        axis=0)
        T1_all_healthy = np.concatenate((T1_ATLAS, T1_NITRC_batch_1, T1_NITRC_batch_2, T1_OA, T1_MIMS, T1_BMI, T1_NOVA),
                                        axis=0)
        age_all_healthy = np.concatenate(
            (age_ATLAS, age_NITRC_batch_1, age_NITRC_batch_2, age_OA, age_MIMS, age_BMI, age_NOVA), axis=0)
        sex_all_healthy = np.concatenate(
            (sex_ATLAS, sex_NITRC_batch_1, sex_NITRC_batch_2, sex_OA, sex_MIMS, sex_BMI, sex_NOVA), axis=0)
        study_all_healthy = np.concatenate(
            (study_ATLAS, study_NITRC_batch_1, study_NITRC_batch_2, study_OA, study_MIMS, study_BMI, study_NOVA),
            axis=0)
        id_all_healthy = np.concatenate((id_ATLAS, id_NITRC_batch_1, id_NITRC_batch_2, id_OA, id_MIMS, id_BMI, id_NOVA),
                                        axis=0)

        return (
        stiffness_all_healthy, dr_all_healthy, T1_all_healthy, age_all_healthy, sex_all_healthy, study_all_healthy,
        id_all_healthy)

    elif study == 'ATLAS+NITRC1+NITRC2+MIMS+BMI+NOVA':
        stiffness = np.concatenate((stiffness_ATLAS, stiffness_NITRC_batch_1, stiffness_NITRC_batch_2, stiffness_MIMS,
                                    stiffness_BMI, stiffness_NOVA), axis=0)
        dr = np.concatenate((dr_ATLAS, dr_NITRC_batch_1, dr_NITRC_batch_2, dr_MIMS, dr_BMI, dr_NOVA), axis=0)
        T1 = np.concatenate((T1_ATLAS, T1_NITRC_batch_1, T1_NITRC_batch_2, T1_MIMS, T1_BMI, T1_NOVA), axis=0)
        age = np.concatenate((age_ATLAS, age_NITRC_batch_1, age_NITRC_batch_2, age_MIMS, age_BMI, age_NOVA), axis=0)
        sex = np.concatenate((sex_ATLAS, sex_NITRC_batch_1, sex_NITRC_batch_2, sex_MIMS, sex_BMI, sex_NOVA), axis=0)
        study = np.concatenate(
            (study_ATLAS, study_NITRC_batch_1, study_NITRC_batch_2, study_MIMS, study_BMI, study_NOVA),
            axis=0)
        id = np.concatenate((id_ATLAS, id_NITRC_batch_1, id_NITRC_batch_2, id_MIMS, id_BMI, id_NOVA), axis=0)

        return (stiffness, dr, T1, age, sex, study, id)

    elif study == 'AD':
        stiffness = stiffness_AD
        dr = dr_AD
        T1 = T1_AD
        age = age_AD
        sex = sex_AD
        study = study_AD
        id = id_AD

        return (stiffness, dr, T1, age, sex, study, id)

    elif study == 'OA':
        stiffness = stiffness_OA
        dr = dr_OA
        T1 = T1_OA
        age = age_OA
        sex = sex_OA
        study = study_OA
        id = id_OA

        return (stiffness, dr, T1, age, sex, study, id)

    elif study == 'ATLAS+NITRC1+NITRC2+OA+BMI+NOVA':
        stiffness = np.concatenate((stiffness_ATLAS, stiffness_NITRC_batch_1, stiffness_NITRC_batch_2, stiffness_OA,
                                    stiffness_BMI, stiffness_NOVA), axis=0)
        dr = np.concatenate((dr_ATLAS, dr_NITRC_batch_1, dr_NITRC_batch_2, dr_OA, dr_BMI, dr_NOVA), axis=0)
        T1 = np.concatenate((T1_ATLAS, T1_NITRC_batch_1, T1_NITRC_batch_2, T1_OA, T1_BMI, T1_NOVA), axis=0)
        age = np.concatenate((age_ATLAS, age_NITRC_batch_1, age_NITRC_batch_2, age_OA, age_BMI, age_NOVA), axis=0)
        sex = np.concatenate((sex_ATLAS, sex_NITRC_batch_1, sex_NITRC_batch_2, sex_OA, sex_BMI, sex_NOVA), axis=0)
        study = np.concatenate((study_ATLAS, study_NITRC_batch_1, study_NITRC_batch_2, study_OA, study_BMI, study_NOVA),
                               axis=0)
        id = np.concatenate((id_ATLAS, id_NITRC_batch_1, id_NITRC_batch_2, id_OA, id_BMI, id_NOVA), axis=0)

        return (stiffness, dr, T1, age, sex, study, id)

    elif study == 'MCI':
        stiffness = stiffness_MIMS_MCI
        dr = dr_MIMS_MCI
        T1 = T1_MIMS_MCI
        age = age_MIMS_MCI
        sex = sex_MIMS_MCI
        study = study_MIMS_MCI
        id = id_MIMS_MCI

        return (stiffness, dr, T1, age, sex, study, id)

    elif study == 'MIMS':
        stiffness = stiffness_MIMS
        dr = dr_MIMS
        T1 = T1_MIMS
        age = age_MIMS
        sex = sex_MIMS
        study = study_MIMS
        id = id_MIMS

        return (stiffness, dr, T1, age, sex, study, id)


def extract_number(filename):
    match = re.search(r'_(\d+)_', filename)
    if match:
        return match.group(1)
    else:
        return None


def extract_second_number(filename):
    match = re.search(r'_(\d+)_.*_(\d+)_', filename)
    if match:
        return match.group(2)
    else:
        return None


def load_region(study, region, location):
    if location == 'local':
        prefix_path = '/Users/jakobtraeuble/PycharmProjects/BrainAgeMRE/Data'

    elif location == 'cluster':
        prefix_path = '/home/jnt27/rds/hpc-work/MRE'

    else:
        raise ValueError('specific location as local or cluster')

    if region == 'all':
        region_labels = []
        for region, label in subcortical_structures.items():
            region_labels.extend(label)

    elif region in subcortical_structures.keys():
        region_labels = subcortical_structures[region]

    else:
        raise ValueError('Invalid region name. Please choose from: all, ' + ', '.join(subcortical_structures.keys()))

    ### ATLAS STUDY

    segm_ATLAS_dir = prefix_path + '/Lucy_Hiscox/MRE_atlas/ATLAS_resampled_segmentations_files/'
    stiffness_ATLAS_dir = prefix_path + '/Lucy_Hiscox/MRE_atlas/ATLAS_individual_stiffness_files/'
    dr_ATLAS_dir = prefix_path + '/Lucy_Hiscox/MRE_atlas/ATLAS_individual_damping_ratio_files/'

    demographics_ATLAS = pd.read_excel(prefix_path + '/Lucy_Hiscox/MRE_atlas/MRE134_Demographics.xlsx')

    stiffness_regions_ATLAS = []
    dr_regions_ATLAS = []

    age_ATLAS = []
    ID_ATLAS = []

    for file in os.listdir(segm_ATLAS_dir):
        if not file.endswith('.nii'):
            continue

        number = extract_number(file)

        segm_img = nib.load(os.path.join(segm_ATLAS_dir, file))
        segm_data = segm_img.get_fdata()

        stiffness_img = nib.load(os.path.join(stiffness_ATLAS_dir, number + 'Stiffness_warped.nii'))
        stiffness_data = stiffness_img.get_fdata()

        dr_img = nib.load(os.path.join(dr_ATLAS_dir, number + 'DR_warped.nii'))
        dr_data = dr_img.get_fdata()

        mask = create_combined_mask(segm_data, region_labels)

        masked_stiffness = apply_mask(stiffness_data, mask)
        masked_dr = apply_mask(dr_data, mask)

        stiffness_regions_ATLAS.append(masked_stiffness)
        dr_regions_ATLAS.append(masked_dr)

        age_ATLAS.append(demographics_ATLAS[demographics_ATLAS['SUBJECT ID'] == int(number)]['AGE'].values[0])
        ID_ATLAS.append('ATLAS_' + number)

    stiffness_regions_ATLAS = np.array(stiffness_regions_ATLAS)
    dr_regions_ATLAS = np.array(dr_regions_ATLAS)
    age_ATLAS = np.array(age_ATLAS)
    ID_ATLAS = np.array(ID_ATLAS)

    ### OA STUDY

    segm_OA_dir = prefix_path + '/Lucy_Hiscox/AD_CN/OA_resampled_segmentations_files/'
    stiffness_OA_dir = prefix_path + '/Lucy_Hiscox/AD_CN/OA_individual_stiffness_files/'
    dr_OA_dir = prefix_path + '/Lucy_Hiscox/AD_CN/OA_individual_damping_ratio_files/'

    demographics_OA = pd.read_excel(prefix_path + '/Lucy_Hiscox/AD_CN/demographics_complete.xlsx')

    stiffness_regions_OA = []
    dr_regions_OA = []

    age_OA = []
    ID_OA = []

    for file in os.listdir(segm_OA_dir):
        if not file.endswith('.nii'):
            continue

        number = extract_number(file)

        segm_img = nib.load(os.path.join(segm_OA_dir, file))
        segm_data = segm_img.get_fdata()

        stiffness_img = nib.load(os.path.join(stiffness_OA_dir, number + '_Stiffness2MNI.nii'))
        stiffness_data = stiffness_img.get_fdata()

        dr_img = nib.load(os.path.join(dr_OA_dir, number + '_DR2MNI.nii'))
        dr_data = dr_img.get_fdata()

        mask = create_combined_mask(segm_data, region_labels)

        masked_stiffness = apply_mask(stiffness_data, mask)
        masked_dr = apply_mask(dr_data, mask)

        stiffness_regions_OA.append(masked_stiffness)
        dr_regions_OA.append(masked_dr)

        age_OA.append(demographics_OA[demographics_OA['Participant ID'] == int(number)]['Age'].values[0])
        ID_OA.append('OA_' + number)

    stiffness_regions_OA = np.array(stiffness_regions_OA)
    dr_regions_OA = np.array(dr_regions_OA)
    age_OA = np.array(age_OA)
    ID_OA = np.array(ID_OA)

    ### OA STUDY

    segm_AD_dir = prefix_path + '/Lucy_Hiscox/AD_CN/AD_resampled_segmentations_files/'
    stiffness_AD_dir = prefix_path + '/Lucy_Hiscox/AD_CN/AD_individual_stiffness_files/'
    dr_AD_dir = prefix_path + '/Lucy_Hiscox/AD_CN/AD_individual_damping_ratio_files/'

    demographics_AD = pd.read_excel(prefix_path + '/Lucy_Hiscox/AD_CN/demographics_complete.xlsx')

    stiffness_regions_AD = []
    dr_regions_AD = []

    age_AD = []
    ID_AD = []

    for file in os.listdir(segm_AD_dir):
        if not file.endswith('.nii'):
            continue

        number = extract_number(file)

        segm_img = nib.load(os.path.join(segm_AD_dir, file))
        segm_data = segm_img.get_fdata()

        stiffness_img = nib.load(os.path.join(stiffness_AD_dir, number + '_Stiffness2MNI.nii'))
        stiffness_data = stiffness_img.get_fdata()

        dr_img = nib.load(os.path.join(dr_AD_dir, number + '_DR2MNI.nii'))
        dr_data = dr_img.get_fdata()

        mask = create_combined_mask(segm_data, region_labels)

        masked_stiffness = apply_mask(stiffness_data, mask)
        masked_dr = apply_mask(dr_data, mask)

        stiffness_regions_AD.append(masked_stiffness)
        dr_regions_AD.append(masked_dr)

        age_AD.append(demographics_AD[demographics_AD['Participant ID'] == int(number)]['Age'].values[0])
        ID_AD.append('AD_' + number)

    stiffness_regions_AD = np.array(stiffness_regions_AD)
    dr_regions_AD = np.array(dr_regions_AD)
    age_AD = np.array(age_AD)
    ID_AD = np.array(ID_AD)

    ### MIMS STUDY

    segm_MIMS_dir = prefix_path + '/Curtis_Johnson/MIMS/MIMS_resampled_segmentations_files/'
    stiffness_MIMS_dir = prefix_path + '/Curtis_Johnson/MIMS/MIMS_individual_stiffness_files/'
    dr_MIMS_dir = prefix_path + '/Curtis_Johnson/MIMS/MIMS_individual_damping_ratio_files/'

    demographics_MIMS = pd.read_excel(prefix_path + '/Curtis_Johnson/MIMS/MIMS_demographics.xlsx')
    demographics_MIMS['Subject ID'] = demographics_MIMS['Subject ID'].str.strip("'\"")

    stiffness_regions_MIMS = []
    dr_regions_MIMS = []

    age_MIMS = []
    ID_MIMS = []

    for file in os.listdir(segm_MIMS_dir):

        if not file.endswith('.nii'):
            continue

        number = extract_number(file)

        segm_img = nib.load(os.path.join(segm_MIMS_dir, file))
        segm_data = segm_img.get_fdata()

        stiffness_img = nib.load(os.path.join(stiffness_MIMS_dir, number + '_stiffness2MNI_warped_v2.nii'))
        stiffness_data = stiffness_img.get_fdata()

        dr_img = nib.load(os.path.join(dr_MIMS_dir, number + '_DR2MNI_warped_v2.nii'))
        dr_data = dr_img.get_fdata()

        mask = create_combined_mask(segm_data, region_labels)

        masked_stiffness = apply_mask(stiffness_data, mask)
        masked_dr = apply_mask(dr_data, mask)

        stiffness_regions_MIMS.append(masked_stiffness)
        dr_regions_MIMS.append(masked_dr)

        demo_name = 'MIMS_HC_' + number

        age_MIMS.append(demographics_MIMS[demographics_MIMS['Subject ID'] == demo_name]['Age (years)'].values[0])
        ID_MIMS.append('MIMS_' + number)

    stiffness_regions_MIMS = np.array(stiffness_regions_MIMS)
    dr_regions_MIMS = np.array(dr_regions_MIMS)
    age_MIMS = np.array(age_MIMS)
    ID_MIMS = np.array(ID_MIMS)

    ### MIMS_MCI STUDY

    segm_MIMS_MCI_dir = prefix_path + '/Curtis_Johnson/MIMS_MCI/MIMS_MCI_resampled_segmentations_files/'
    stiffness_MIMS_MCI_dir = prefix_path + '/Curtis_Johnson/MIMS_MCI/MIMS_MCI_individual_stiffness_files/'
    dr_MIMS_MCI_dir = prefix_path + '/Curtis_Johnson/MIMS_MCI/MIMS_MCI_individual_damping_ratio_files/'

    demographics_MIMS_MCI = pd.read_excel(prefix_path + '/Curtis_Johnson/MIMS_MCI/MIMS_MCI.xlsx')

    stiffness_regions_MIMS_MCI = []
    dr_regions_MIMS_MCI = []

    age_MIMS_MCI = []
    ID_MIMS_MCI = []

    for file in os.listdir(segm_MIMS_MCI_dir):
        if not file.endswith('.nii'):
            continue

        number = extract_number(file)

        segm_img = nib.load(os.path.join(segm_MIMS_MCI_dir, file))
        segm_data = segm_img.get_fdata()

        stiffness_img = nib.load(os.path.join(stiffness_MIMS_MCI_dir, number + '_stiffness2MNI_warped.nii'))
        stiffness_data = stiffness_img.get_fdata()

        dr_img = nib.load(os.path.join(dr_MIMS_MCI_dir, number + '_DR2MNI_warped.nii'))
        dr_data = dr_img.get_fdata()

        mask = create_combined_mask(segm_data, region_labels)

        masked_stiffness = apply_mask(stiffness_data, mask)
        masked_dr = apply_mask(dr_data, mask)

        stiffness_regions_MIMS_MCI.append(masked_stiffness)
        dr_regions_MIMS_MCI.append(masked_dr)

        demo_name = 'MCI' + number

        age_MIMS_MCI.append(demographics_MIMS_MCI[demographics_MIMS_MCI['ID'] == demo_name]['Age'].values[0])
        ID_MIMS_MCI.append('MIMS_MCI_' + number)

    stiffness_regions_MIMS_MCI = np.array(stiffness_regions_MIMS_MCI)
    dr_regions_MIMS_MCI = np.array(dr_regions_MIMS_MCI)
    age_MIMS_MCI = np.array(age_MIMS_MCI)
    ID_MIMS_MCI = np.array(ID_MIMS_MCI)

    ### BMI STUDY

    segm_BMI_dir = prefix_path + '/Curtis_Johnson/BMI/BMI_resampled_segmentations_files/'
    stiffness_BMI_dir = prefix_path + '/Curtis_Johnson/BMI/BMI_individual_stiffness_files/'
    dr_BMI_dir = prefix_path + '/Curtis_Johnson/BMI/BMI_individual_damping_ratio_files/'

    demographics_BMI = pd.read_excel(prefix_path + '/Curtis_Johnson/BMI/NOVAandBMI.xlsx')

    stiffness_regions_BMI = []
    dr_regions_BMI = []

    age_BMI = []
    ID_BMI = []

    for file in os.listdir(segm_BMI_dir):
        if not file.endswith('.nii'):
            continue

        number = extract_number(file)

        segm_img = nib.load(os.path.join(segm_BMI_dir, file))
        segm_data = segm_img.get_fdata()

        stiffness_img = nib.load(os.path.join(stiffness_BMI_dir, number + '_stiffness2MNI_warped.nii'))
        stiffness_data = stiffness_img.get_fdata()

        dr_img = nib.load(os.path.join(dr_BMI_dir, number + '_DR2MNI_warped.nii'))
        dr_data = dr_img.get_fdata()

        mask = create_combined_mask(segm_data, region_labels)

        masked_stiffness = apply_mask(stiffness_data, mask)
        masked_dr = apply_mask(dr_data, mask)

        stiffness_regions_BMI.append(masked_stiffness)
        dr_regions_BMI.append(masked_dr)

        demo_name = 'BMI_' + number

        age_BMI.append(demographics_BMI[demographics_BMI['ID'] == demo_name]['Age'].values[0])
        ID_BMI.append('BMI_' + number)

    stiffness_regions_BMI = np.array(stiffness_regions_BMI)
    dr_regions_BMI = np.array(dr_regions_BMI)
    age_BMI = np.array(age_BMI)
    ID_BMI = np.array(ID_BMI)

    ### NOVA STUDY

    segm_NOVA_dir = prefix_path + '/Curtis_Johnson/NOVA/NOVA_resampled_segmentations_files/'
    stiffness_NOVA_dir = prefix_path + '/Curtis_Johnson/NOVA/NOVA_individual_stiffness_files/'
    dr_NOVA_dir = prefix_path + '/Curtis_Johnson/NOVA/NOVA_individual_damping_ratio_files/'

    demographics_NOVA = pd.read_excel(prefix_path + '/Curtis_Johnson/BMI/NOVAandBMI.xlsx')

    stiffness_regions_NOVA = []
    dr_regions_NOVA = []

    age_NOVA = []
    ID_NOVA = []

    for file in os.listdir(segm_NOVA_dir):
        if not file.endswith('.nii'):
            continue

        number = extract_number(file)

        segm_img = nib.load(os.path.join(segm_NOVA_dir, file))
        segm_data = segm_img.get_fdata()

        stiffness_img = nib.load(os.path.join(stiffness_NOVA_dir, number + '_stiffness2MNI_warped.nii'))
        stiffness_data = stiffness_img.get_fdata()

        dr_img = nib.load(os.path.join(dr_NOVA_dir, number + '_DR2MNI_warped.nii'))
        dr_data = dr_img.get_fdata()

        mask = create_combined_mask(segm_data, region_labels)

        masked_stiffness = apply_mask(stiffness_data, mask)
        masked_dr = apply_mask(dr_data, mask)

        stiffness_regions_NOVA.append(masked_stiffness)
        dr_regions_NOVA.append(masked_dr)

        demo_name = 'NOVA_' + number

        age_NOVA.append(demographics_NOVA[demographics_NOVA['ID'] == demo_name]['Age'].values[0])
        ID_NOVA.append('NOVA_' + number)

    stiffness_regions_NOVA = np.array(stiffness_regions_NOVA)
    dr_regions_NOVA = np.array(dr_regions_NOVA)
    age_NOVA = np.array(age_NOVA)
    ID_NOVA = np.array(ID_NOVA)

    ### NITRC BATCH 1 STUDY

    segm_NITRC_batch_1_dir = prefix_path + '/Curtis_Johnson/NITRC_batch_1/NITRC_batch_1_resampled_segmentations_files/'
    stiffness_NITRC_batch_1_dir = prefix_path + '/Curtis_Johnson/NITRC_batch_1/NITRC_batch_1_individual_stiffness_files/'
    dr_NITRC_batch_1_dir = prefix_path + '/Curtis_Johnson/NITRC_batch_1/NITRC_batch_1_individual_damping_ratio_files/'

    demographics_NITRC_batch_1 = pd.read_excel(
        prefix_path + '/Curtis_Johnson/NITRC_batch_1/U01_NITRC_subject_info_v03a.xlsx')

    stiffness_regions_NITRC_batch_1 = []
    dr_regions_NITRC_batch_1 = []

    age_NITRC_batch_1 = []
    ID_NITRC_batch_1 = []

    for file in os.listdir(segm_NITRC_batch_1_dir):
        if not file.endswith('.nii'):
            continue

        number = extract_second_number(file).zfill(2)

        segm_img = nib.load(os.path.join(segm_NITRC_batch_1_dir, file))
        segm_data = segm_img.get_fdata()

        stiffness_img = nib.load(os.path.join(stiffness_NITRC_batch_1_dir, number + '_stiffness2MNI_warped.nii'))
        stiffness_data = stiffness_img.get_fdata()

        dr_img = nib.load(os.path.join(dr_NITRC_batch_1_dir, number + '_dr2MNI_warped.nii'))
        dr_data = dr_img.get_fdata()

        mask = create_combined_mask(segm_data, region_labels)

        masked_stiffness = apply_mask(stiffness_data, mask)
        masked_dr = apply_mask(dr_data, mask)

        stiffness_regions_NITRC_batch_1.append(masked_stiffness)
        dr_regions_NITRC_batch_1.append(masked_dr)

        demo_name = 'U01_UDEL_00' + number

        age_NITRC_batch_1.append(
            demographics_NITRC_batch_1[demographics_NITRC_batch_1['Subject'] == demo_name]['Age (yr)'].values[0])
        ID_NITRC_batch_1.append('NITRC_batch_1_' + number)

    stiffness_regions_NITRC_batch_1 = np.array(stiffness_regions_NITRC_batch_1)
    dr_regions_NITRC_batch_1 = np.array(dr_regions_NITRC_batch_1)
    age_NITRC_batch_1 = np.array(age_NITRC_batch_1)
    ID_NITRC_batch_1 = np.array(ID_NITRC_batch_1)

    ### NITRC BATCH 2 STUDY

    segm_NITRC_batch_2_dir = prefix_path + '/Curtis_Johnson/NITRC_batch_2/NITRC_batch_2_resampled_segmentations_files/'
    stiffness_NITRC_batch_2_dir = prefix_path + '/Curtis_Johnson/NITRC_batch_2/NITRC_batch_2_individual_stiffness_files/'
    dr_NITRC_batch_2_dir = prefix_path + '/Curtis_Johnson/NITRC_batch_2/NITRC_batch_2_individual_damping_ratio_files/'

    stiffness_regions_NITRC_batch_2 = []
    dr_regions_NITRC_batch_2 = []

    age_NITRC_batch_2 = []
    ID_NITRC_batch_2 = []

    for file in os.listdir(segm_NITRC_batch_2_dir):
        if not file.endswith('.nii'):
            continue

        number = extract_second_number(file)[2:]

        segm_img = nib.load(os.path.join(segm_NITRC_batch_2_dir, file))
        segm_data = segm_img.get_fdata()

        stiffness_img = nib.load(os.path.join(stiffness_NITRC_batch_2_dir, number + '_stiffness2MNI_warped.nii'))
        stiffness_data = stiffness_img.get_fdata()

        dr_img = nib.load(os.path.join(dr_NITRC_batch_2_dir, number + '_DR2MNI_warped.nii'))
        dr_data = dr_img.get_fdata()

        mask = create_combined_mask(segm_data, region_labels)

        masked_stiffness = apply_mask(stiffness_data, mask)
        masked_dr = apply_mask(dr_data, mask)

        stiffness_regions_NITRC_batch_2.append(masked_stiffness)
        dr_regions_NITRC_batch_2.append(masked_dr)

        mat = scipy.io.loadmat(
            prefix_path + '/Curtis_Johnson/NITRC_batch_2/NITRC_batch_2_individual_subj_info/' + number.zfill(
                2) + '_1_subjInfo.mat')

        age_NITRC_batch_2.append(mat['subjInfo']['subjAge'][0][0][0][0])
        ID_NITRC_batch_2.append('NITRC_batch_2_' + number)

    stiffness_regions_NITRC_batch_2 = np.array(stiffness_regions_NITRC_batch_2)
    dr_regions_NITRC_batch_2 = np.array(dr_regions_NITRC_batch_2)
    age_NITRC_batch_2 = np.array(age_NITRC_batch_2)
    ID_NITRC_batch_2 = np.array(ID_NITRC_batch_2)

    if study == 'ATLAS+NITRC1+NITRC2+OA+MIMS+BMI+NOVA':
        stiffness = np.concatenate((stiffness_regions_ATLAS, stiffness_regions_NITRC_batch_1,
                                    stiffness_regions_NITRC_batch_2, stiffness_regions_OA,
                                    stiffness_regions_MIMS, stiffness_regions_BMI, stiffness_regions_NOVA), axis=0)
        dr = np.concatenate((dr_regions_ATLAS, dr_regions_NITRC_batch_1, dr_regions_NITRC_batch_2, dr_regions_OA,
                             dr_regions_MIMS, dr_regions_BMI, dr_regions_NOVA), axis=0)
        age = np.concatenate((age_ATLAS, age_NITRC_batch_1, age_NITRC_batch_2, age_OA, age_MIMS, age_BMI, age_NOVA),
                             axis=0)
        ID = np.concatenate((ID_ATLAS, ID_NITRC_batch_1, ID_NITRC_batch_2, ID_OA, ID_MIMS, ID_BMI, ID_NOVA), axis=0)

        stiffness = stiffness[:, 15:75, 15:80, 0:60]
        dr = dr[:, 15:75, 15:80, 0:60]

        return (stiffness, dr, age, ID)

    elif study == 'ATLAS+NITRC1+NITRC2+MIMS+BMI+NOVA':
        stiffness = np.concatenate((stiffness_regions_ATLAS, stiffness_regions_NITRC_batch_1,
                                    stiffness_regions_NITRC_batch_2, stiffness_regions_MIMS, stiffness_regions_BMI,
                                    stiffness_regions_NOVA), axis=0)
        dr = np.concatenate((dr_regions_ATLAS, dr_regions_NITRC_batch_1, dr_regions_NITRC_batch_2, dr_regions_MIMS,
                             dr_regions_BMI, dr_regions_NOVA), axis=0)
        age = np.concatenate((age_ATLAS, age_NITRC_batch_1, age_NITRC_batch_2, age_MIMS, age_BMI, age_NOVA), axis=0)
        ID = np.concatenate((ID_ATLAS, ID_NITRC_batch_1, ID_NITRC_batch_2, ID_MIMS, ID_BMI, ID_NOVA), axis=0)

        stiffness = stiffness[:, 15:75, 15:80, 0:60]
        dr = dr[:, 15:75, 15:80, 0:60]

        return (stiffness, dr, age, ID)

    elif study == 'AD':
        stiffness = stiffness_regions_AD
        dr = dr_regions_AD
        age = age_AD
        ID = ID_AD

        stiffness = stiffness[:, 15:75, 15:80, 0:60]
        dr = dr[:, 15:75, 15:80, 0:60]

        return (stiffness, dr, age, ID)

    elif study == 'OA':
        stiffness = stiffness_regions_OA
        dr = dr_regions_OA
        age = age_OA
        ID = ID_OA

        stiffness = stiffness[:, 15:75, 15:80, 0:60]
        dr = dr[:, 15:75, 15:80, 0:60]

        return (stiffness, dr, age, ID)

    elif study == 'ATLAS+NITRC1+NITRC2+OA+BMI+NOVA':
        stiffness = np.concatenate((stiffness_regions_ATLAS, stiffness_regions_NITRC_batch_1,
                                    stiffness_regions_NITRC_batch_2, stiffness_regions_OA, stiffness_regions_BMI,
                                    stiffness_regions_NOVA), axis=0)
        dr = np.concatenate((dr_regions_ATLAS, dr_regions_NITRC_batch_1, dr_regions_NITRC_batch_2, dr_regions_OA,
                             dr_regions_BMI, dr_regions_NOVA), axis=0)
        age = np.concatenate((age_ATLAS, age_NITRC_batch_1, age_NITRC_batch_2, age_OA, age_BMI, age_NOVA), axis=0)
        ID = np.concatenate((ID_ATLAS, ID_NITRC_batch_1, ID_NITRC_batch_2, ID_OA, ID_BMI, ID_NOVA), axis=0)

        stiffness = stiffness[:, 15:75, 15:80, 0:60]
        dr = dr[:, 15:75, 15:80, 0:60]

        return (stiffness, dr, age, ID)

    elif study == 'MCI':
        stiffness = stiffness_regions_MIMS_MCI
        dr = dr_regions_MIMS_MCI
        age = age_MIMS_MCI
        ID = ID_MIMS_MCI

        stiffness = stiffness[:, 15:75, 15:80, 0:60]
        dr = dr[:, 15:75, 15:80, 0:60]

        return (stiffness, dr, age, ID)

    elif study == 'MIMS':
        stiffness = stiffness_regions_MIMS
        dr = dr_regions_MIMS
        age = age_MIMS
        ID = ID_MIMS

        stiffness = stiffness[:, 15:75, 15:80, 0:60]
        dr = dr[:, 15:75, 15:80, 0:60]

        return (stiffness, dr, age, ID)


label_dict = {
    0: 'Background',
    2: 'Left-Cerebral-White-Matter',
    3: 'Left-Cerebral-Cortex',
    4: 'Left-Lateral-Ventricle',
    5: 'Left-Inf-Lat-Vent',
    7: 'Left-Cerebellum-White-Matter',
    8: 'Left-Cerebellum-Cortex',
    10: 'Left-Thalamus-Proper',
    11: 'Left-Caudate',
    12: 'Left-Putamen',
    13: 'Left-Pallidum',
    14: '3rd-Ventricle',
    15: '4th-Ventricle',
    16: 'Brain-Stem',
    17: 'Left-Hippocampus',
    18: 'Left-Amygdala',
    24: 'CSF',
    26: 'Left-Accumbens-area',
    28: 'Left-VentralDC',
    31: 'Left-choroid-plexus',
    41: 'Right-Cerebral-White-Matter',
    42: 'Right-Cerebral-Cortex',
    43: 'Right-Lateral-Ventricle',
    44: 'Right-Inf-Lat-Vent',
    46: 'Right-Cerebellum-White-Matter',
    47: 'Right-Cerebellum-Cortex',
    49: 'Right-Thalamus-Proper',
    50: 'Right-Caudate',
    51: 'Right-Putamen',
    52: 'Right-Pallidum',
    53: 'Right-Hippocampus',
    54: 'Right-Amygdala',
    58: 'Right-Accumbens-area',
    60: 'Right-VentralDC',
    63: 'Right-choroid-plexus',
    77: 'WM-hypointensities'
}

# Combined regions mapping
combined_regions = {
    'Cerebral-White-Matter': [2, 41],
    'Cerebral-Cortex': [3, 42],
    'Lateral-Ventricle': [4, 43],
    'Inf-Lat-Vent': [5, 44],
    'Cerebellum-White-Matter': [7, 46],
    'Cerebellum-Cortex': [8, 47],
    'Thalamus-Proper': [10, 49],
    'Caudate': [11, 50],
    'Putamen': [12, 51],
    'Pallidum': [13, 52],
    'Hippocampus': [17, 53],
    'Amygdala': [18, 54],
    'Accumbens-area': [26, 58],
    'VentralDC': [28, 60],
    'Choroid-plexus': [31, 63]
}

subcortical_structures = {'Accumbens-area': combined_regions['Accumbens-area'],
                          'Amygdala': combined_regions['Amygdala'],
                          'Brain-Stem': [16],
                          'Caudate': combined_regions['Caudate'],
                          'Cerebellum': [7, 8, 46, 47],
                          'VentralDC': combined_regions['VentralDC'],
                          'Hippocampus': combined_regions['Hippocampus'],
                          'Pallidum': combined_regions['Pallidum'],
                          'Putamen': combined_regions['Putamen'],
                          'Thalamus-Proper': combined_regions['Thalamus-Proper']}


# Function to create a mask for a given set of labels
def create_combined_mask(data, labels):
    mask = np.isin(data, labels)
    return mask


# Function to apply a mask to an image
def apply_mask(image, mask):
    return image * mask