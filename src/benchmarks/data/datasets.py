import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import convolve1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from scipy.ndimage import rotate
from scipy.stats import norm
from sklearn.model_selection import KFold
from torchvision import transforms

class MREDataset(Dataset):

  def __init__(self, modality, train, soft_sigma, lds, noise_std, location, flatten, fold):

    if soft_sigma not in [1, 2, None]:
      raise ValueError("'{}' is not a valid soft_sigma. Valid soft_sigma is 1, 2 or None!".format(soft_sigma))

    self.soft_sigma = soft_sigma

    if noise_std is not None:
      self.T = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x).float()),
            AddGaussianNoiseTensor(mean=0., std=noise_std)
        ])
    else:
      self.T = transforms.Compose([
            transforms.Lambda(lambda x: torch.tensor(x).float())
        ])

    if lds:
        reweight = 'inverse'
    else:
        reweight = 'none'

    stiffness, dr, T1, age, sex, study, id = load_samples(study='ATLAS+NITRC1+NITRC2+OA+MIMS+BMI+NOVA',
                                                          location=location, flatten=flatten)

    self.age_integers = age
    self.age = age
    self.weights = self._prepare_weights(lds=lds, reweight=reweight)
    sex = np.array([categorical_one_hot_encoding(category='sex', label=i) for i in sex])
    study = np.array([categorical_one_hot_encoding(category='study', label=i) for i in study])

    if soft_sigma in [1,2]:
        bin_step = 1
        bin_range = [1 - (bin_step / 2), 100 + (bin_step / 2)]
        self.age, bin_centers = num2vect(x=self.age, bin_range=bin_range, bin_step=bin_step, sigma=soft_sigma)

    if modality == 'stiffness':
        _, mu_stiff, sigma_stiff = normalize_mean_0_std_1(stiffness, default_value=0, mu_nonzero=None, sigma_nonzero=None)
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

    assert fold in range(5)

    if fold in range(5):
        for fold_iter, (train_ids, test_ids) in enumerate(kfold.split(stiffness)):

            if fold_iter == fold:
                stiffness_train, stiffness_test = stiffness[train_ids], stiffness[test_ids]
                dr_train, dr_test = dr[train_ids], dr[test_ids]
                T1_train, T1_test = T1[train_ids], T1[test_ids]
                sex_train, sex_test = sex[train_ids], sex[test_ids]
                age_train, age_test = self.age[train_ids], self.age[test_ids]
                study_train, study_test = study[train_ids], study[test_ids]

            else:
                continue

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

  def _prepare_weights(self, lds=False, reweight='none', lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
    assert reweight in {'none', 'inverse', 'sqrt_inv'}
    assert reweight != 'none' if lds else True, \
      "Set reweight to \'sqrt_inv\' (default) or \'inverse\' when using LDS"

    if reweight == 'none':
      return None

    max_target = int(np.max(self.age_integers) + 1)

    value_dict = {x: 0 for x in range(max_target)}
    labels = self.age_integers.tolist()
    # mbr
    for label in labels:
      value_dict[min(max_target - 1, int(label))] += 1
    if reweight == 'sqrt_inv':
      value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
    elif reweight == 'inverse':
      value_dict = {k: np.clip(v, 1, 1000) for k, v in value_dict.items()}  # clip weights for inverse re-weight
    num_per_label = [value_dict[min(max_target - 1, int(label))] for label in labels]
    if not len(num_per_label) or reweight == 'none':
      return None

    if lds:
      lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
      smoothed_value = convolve1d(
        np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')
      num_per_label = [smoothed_value[min(max_target - 1, int(label))] for label in labels]

    weights = [np.float32(1 / x) for x in num_per_label]
    scaling = len(weights) / np.sum(weights)
    weights = [scaling * x for x in weights]
    return weights

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

      if self.soft_sigma in [1, 2]:
        y = torch.from_numpy(self.y[index]).to(torch.float32)
      else:
        y = np.array([self.y[index]]).astype(np.float32)  # Convert scalar to array
        y = torch.from_numpy(y)  # Convert to tensor


      sex = torch.from_numpy(self.sex[index]).to(torch.float32)
      site = torch.from_numpy(self.site[index]).to(torch.float32)
      weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])

      if self.modality == 'dr+stiffness':

          x_dr = self.x_dr[index]
          x_stiff = self.x_stiff[index]

          x_dr = self.T(x_dr)
          x_stiff = self.T(x_stiff)

          return (x_dr.to(torch.float32), x_stiff.to(torch.float32)), y, sex, site, weight

      else:
          x = self.x[index]

          x = self.T(x)

          return x.to(torch.float32), y, sex, site, weight


def load_samples(study, location, flatten):

  if location == 'local':
    prefix_path = '/Users/jakobtraeuble/PycharmProjects/BrainAgeMRE/Data'

  elif location == 'cluster':
    prefix_path = '/home/jnt27/rds/hpc-work/MRE'

  else:
    raise ValueError('specific location as local or cluster')

  stiffness_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/stiffness_134.npy', allow_pickle=True)
  dr_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/dr_134.npy', allow_pickle=True)
  T1_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/T1_masked_ATLAS.npy', allow_pickle=True)
  age_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/age_ATLAS.npy', allow_pickle=True)
  sex_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/sex_ATLAS.npy', allow_pickle=True)
  id_ATLAS = np.load(prefix_path + '/Lucy_Hiscox/MRE_atlas/id_ATLAS.npy', allow_pickle=True)
  study_ATLAS = np.array(['ATLAS']*len(age_ATLAS))

  stiffness_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/stiffness_OA.npy', allow_pickle=True)
  dr_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/dr_OA.npy', allow_pickle=True)
  T1_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/T1_masked_OA.npy', allow_pickle=True)
  age_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/age_OA.npy', allow_pickle=True)
  sex_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/sex_OA.npy', allow_pickle=True)
  id_OA = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/id_OA.npy', allow_pickle=True)
  study_OA = np.array(['CN']*len(age_OA))

  stiffness_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/stiffness_BMI.npy', allow_pickle=True)
  dr_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/dr_BMI.npy', allow_pickle=True)
  T1_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/T1_masked_BMI.npy', allow_pickle=True)
  age_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/age_BMI.npy', allow_pickle=True)
  sex_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/sex_BMI.npy', allow_pickle=True)
  id_BMI = np.load(prefix_path + '/Curtis_Johnson/BMI/id_BMI.npy', allow_pickle=True)
  study_BMI = np.array(['BMI'] * len(age_BMI))

  stiffness_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/stiffness_NOVA.npy', allow_pickle=True)
  dr_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/dr_NOVA.npy', allow_pickle=True)
  T1_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/T1_masked_NOVA.npy', allow_pickle=True)
  age_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/age_NOVA.npy', allow_pickle=True)
  sex_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/sex_NOVA.npy', allow_pickle=True)
  id_NOVA = np.load(prefix_path + '/Curtis_Johnson/NOVA/id_NOVA.npy', allow_pickle=True)
  study_NOVA = np.array(['NOVA'] * len(age_NOVA))

  stiffness_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/stiffness_NITRC_batch_1.npy', allow_pickle=True)
  dr_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/dr_NITRC_batch_1.npy', allow_pickle=True)
  T1_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/T1_masked_NITRC_batch_1.npy', allow_pickle=True)
  age_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/age_NITRC_batch_1.npy', allow_pickle=True)
  sex_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/sex_NITRC_batch_1.npy', allow_pickle=True)
  id_NITRC_batch_1 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_1/id_NITRC_batch_1.npy', allow_pickle=True)
  study_NITRC_batch_1 = np.array(['NITRC_batch_1']*len(age_NITRC_batch_1))

  stiffness_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/stiffness_NITRC_batch_2.npy', allow_pickle=True)
  dr_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/dr_NITRC_batch_2.npy', allow_pickle=True)
  T1_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/T1_masked_NITRC_batch_2.npy', allow_pickle=True)
  age_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/age_NITRC_batch_2.npy', allow_pickle=True)
  sex_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/sex_NITRC_batch_2.npy', allow_pickle=True)
  id_NITRC_batch_2 = np.load(prefix_path + '/Curtis_Johnson/NITRC_batch_2/id_NITRC_batch_2.npy', allow_pickle=True)
  study_NITRC_batch_2 = np.array(['NITRC_batch_2'] * len(age_NITRC_batch_2))

  stiffness_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/stiffness_MIMS.npy', allow_pickle=True)
  dr_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/dr_MIMS.npy', allow_pickle=True)
  T1_MIMS = np.load(prefix_path + '/Curtis_Johnson/MIMS/T1_masked_MIMS.npy', allow_pickle=True)
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
  study_MIMS_MCI = np.array(['MIMS_MCI'] * len(age_MIMS_MCI))

  stiffness_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/stiffness_AD.npy', allow_pickle=True)
  dr_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/dr_AD.npy', allow_pickle=True)
  T1_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/T1_masked_AD.npy', allow_pickle=True)
  age_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/age_AD.npy', allow_pickle=True)
  sex_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/sex_AD.npy', allow_pickle=True)
  id_AD = np.load(prefix_path + '/Lucy_Hiscox/AD_CN/id_AD.npy', allow_pickle=True)
  study_AD = np.array(['AD'] * len(age_AD))

  if study == 'ATLAS':
    if flatten:
      stiffness_ATLAS = stiffness_ATLAS.reshape(stiffness_ATLAS.shape[0],
                                          stiffness_ATLAS.shape[1] * stiffness_ATLAS.shape[2] * stiffness_ATLAS.shape[3])
      dr_ATLAS = dr_ATLAS.reshape(dr_ATLAS.shape[0], dr_ATLAS.shape[1] * dr_ATLAS.shape[2] * dr_ATLAS.shape[3])
      T1_ATLAS = T1_ATLAS.reshape(T1_ATLAS.shape[0], T1_ATLAS.shape[1] * T1_ATLAS.shape[2] * T1_ATLAS.shape[3])

    return stiffness_ATLAS, dr_ATLAS, T1_ATLAS, age_ATLAS, sex_ATLAS, study_ATLAS, id_ATLAS

  elif study == 'OA':
    if flatten:
      stiffness_OA = stiffness_OA.reshape(stiffness_OA.shape[0],
                                                stiffness_OA.shape[1] * stiffness_OA.shape[2] * stiffness_OA.shape[3])
      dr_OA = dr_OA.reshape(dr_OA.shape[0], dr_OA.shape[1] * dr_OA.shape[2] * dr_OA.shape[3])
      T1_OA = T1_OA.reshape(T1_OA.shape[0], T1_OA.shape[1] * T1_OA.shape[2] * T1_OA.shape[3])

    return stiffness_OA, dr_OA, T1_OA, age_OA, sex_OA, study_OA, id_OA

  elif study == 'AD':
    if flatten:
      stiffness_AD = stiffness_AD.reshape(stiffness_AD.shape[0],
                                          stiffness_AD.shape[1] * stiffness_AD.shape[2] * stiffness_AD.shape[3])
      dr_AD = dr_AD.reshape(dr_AD.shape[0], dr_AD.shape[1] * dr_AD.shape[2] * dr_AD.shape[3])
      T1_AD = T1_OA.reshape(T1_AD.shape[0], T1_AD.shape[1] * T1_AD.shape[2] * T1_AD.shape[3])

    return stiffness_AD, dr_AD, T1_AD, age_AD, sex_AD, study_AD, id_AD

  elif study == 'NITRC_batch_1':
    if flatten:
      stiffness_NITRC_batch_1 = stiffness_NITRC_batch_1.reshape(stiffness_NITRC_batch_1.shape[0],
                                                stiffness_NITRC_batch_1.shape[1] * stiffness_NITRC_batch_1.shape[2] *
                                                stiffness_NITRC_batch_1.shape[3])
      dr_NITRC_batch_1 = dr_NITRC_batch_1.reshape(dr_NITRC_batch_1.shape[0], dr_NITRC_batch_1.shape[1] * dr_NITRC_batch_1.shape[2]
                                         * dr_NITRC_batch_1.shape[3])
      T1_NITRC_batch_1 = T1_NITRC_batch_1.reshape(T1_NITRC_batch_1.shape[0], T1_NITRC_batch_1.shape[1]
                                                  * T1_NITRC_batch_1.shape[2] * T1_NITRC_batch_1.shape[3])

    return stiffness_NITRC_batch_1, dr_NITRC_batch_1, T1_NITRC_batch_1, age_NITRC_batch_1, sex_NITRC_batch_1, study_NITRC_batch_1, id_NITRC_batch_1

  elif study == 'NITRC_batch_2':
    if flatten:
      stiffness_NITRC_batch_2 = stiffness_NITRC_batch_2.reshape(stiffness_NITRC_batch_2.shape[0],
                                                stiffness_NITRC_batch_2.shape[1] * stiffness_NITRC_batch_2.shape[2] *
                                                stiffness_NITRC_batch_2.shape[3])
      dr_NITRC_batch_2 = dr_NITRC_batch_2.reshape(dr_NITRC_batch_2.shape[0], dr_NITRC_batch_2.shape[1] * dr_NITRC_batch_2.shape[2]
                                         * dr_NITRC_batch_2.shape[3])
      T1_NITRC_batch_2 = T1_NITRC_batch_2.reshape(T1_NITRC_batch_2.shape[0], T1_NITRC_batch_2.shape[1] * T1_NITRC_batch_2.shape[2]
                                                  * T1_NITRC_batch_2.shape[3])

    return stiffness_NITRC_batch_2, dr_NITRC_batch_2, T1_NITRC_batch_2, age_NITRC_batch_2, sex_NITRC_batch_2, study_NITRC_batch_2, id_NITRC_batch_2

  elif study == 'MIMS':
    if flatten:
      stiffness_MIMS = stiffness_MIMS.reshape(stiffness_MIMS.shape[0], stiffness_MIMS.shape[1] * stiffness_MIMS.shape[2] *
                                                stiffness_MIMS.shape[3])
      dr_MIMS = dr_MIMS.reshape(dr_MIMS.shape[0], dr_MIMS.shape[1] * dr_MIMS.shape[2]* dr_MIMS.shape[3])
      T1_MIMS = T1_MIMS.reshape(T1_MIMS.shape[0], T1_MIMS.shape[1] * T1_MIMS.shape[2] * T1_MIMS.shape[3])

    return stiffness_MIMS, dr_MIMS, T1_MIMS, age_MIMS, sex_MIMS, study_MIMS, id_MIMS

  elif study == 'BMI':
    if flatten:
      stiffness_BMI = stiffness_BMI.reshape(stiffness_BMI.shape[0], stiffness_BMI.shape[1] * stiffness_BMI.shape[2] *
                                                stiffness_BMI.shape[3])
      dr_BMI = dr_BMI.reshape(dr_BMI.shape[0], dr_BMI.shape[1] * dr_BMI.shape[2]* dr_BMI.shape[3])
      T1_BMI = T1_BMI.reshape(T1_BMI.shape[0], T1_BMI.shape[1] * T1_BMI.shape[2] * T1_BMI.shape[3])

    return stiffness_BMI, dr_BMI, T1_BMI, age_BMI, sex_BMI, study_BMI, id_BMI

  elif study == 'NOVA':
    if flatten:
      stiffness_NOVA = stiffness_NOVA.reshape(stiffness_NOVA.shape[0], stiffness_NOVA.shape[1] * stiffness_NOVA.shape[2] *
                                                stiffness_NOVA.shape[3])
      dr_NOVA = dr_NOVA.reshape(dr_NOVA.shape[0], dr_NOVA.shape[1] * dr_NOVA.shape[2]* dr_NOVA.shape[3])
      T1_NOVA = T1_NOVA.reshape(T1_NOVA.shape[0], T1_NOVA.shape[1] * T1_NOVA.shape[2] * T1_NOVA.shape[3])

    return stiffness_NOVA, dr_NOVA, T1_NOVA, age_NOVA, sex_NOVA, study_NOVA, id_NOVA

  elif study == 'MIMS_MCI':
    if flatten:
      stiffness_MIMS_MCI = stiffness_MIMS_MCI.reshape(stiffness_MIMS_MCI.shape[0], stiffness_MIMS_MCI.shape[1] * stiffness_MIMS_MCI.shape[2] *
                                                stiffness_MIMS_MCI.shape[3])
      dr_MIMS_MCI = dr_MIMS_MCI.reshape(dr_MIMS_MCI.shape[0], dr_MIMS_MCI.shape[1] * dr_MIMS_MCI.shape[2]* dr_MIMS_MCI.shape[3])
      T1_MIMS_MCI = T1_MIMS_MCI.reshape(T1_MIMS_MCI.shape[0], T1_MIMS_MCI.shape[1] * T1_MIMS_MCI.shape[2] * T1_MIMS_MCI.shape[3])

    return stiffness_MIMS_MCI, dr_MIMS_MCI, T1_MIMS_MCI, age_MIMS_MCI, sex_MIMS_MCI, study_MIMS_MCI, id_MIMS_MCI

  elif study == 'ATLAS+NITRC1+NITRC2+BMI+NOVA':
    stiffness_no_controls = np.concatenate((stiffness_ATLAS, stiffness_NITRC_batch_1, stiffness_NITRC_batch_2,
                                    stiffness_MIMS), axis=0)
    dr_no_controls = np.concatenate((dr_ATLAS, dr_NITRC_batch_1, dr_NITRC_batch_2, dr_MIMS), axis=0)
    T1_no_controls = np.concatenate((T1_ATLAS, T1_NITRC_batch_1, T1_NITRC_batch_2, T1_MIMS), axis=0)
    age_no_controls = np.concatenate((age_ATLAS, age_NITRC_batch_1, age_NITRC_batch_2, age_MIMS), axis=0)
    sex_no_controls = np.concatenate((sex_ATLAS, sex_NITRC_batch_1, sex_NITRC_batch_2, sex_MIMS), axis=0)
    study_no_controls = np.concatenate((study_ATLAS, study_NITRC_batch_1, study_NITRC_batch_2, study_MIMS), axis=0)
    id_no_controls = np.concatenate((id_ATLAS, id_NITRC_batch_1, id_NITRC_batch_2, id_MIMS), axis=0)

    if flatten:
      stiffness_no_controls = stiffness_no_controls.reshape(stiffness_no_controls.shape[0],
                                            stiffness_no_controls.shape[1] * stiffness_no_controls.shape[2] * stiffness_no_controls.shape[3])

      dr_no_controls = dr_no_controls.reshape(dr_no_controls.shape[0], dr_no_controls.shape[1] * dr_no_controls.shape[2] * dr_no_controls.shape[3])

      T1_no_controls = T1_no_controls.reshape(T1_no_controls.shape[0], T1_no_controls.shape[1] * T1_no_controls.shape[2] * T1_no_controls.shape[3])


    return stiffness_no_controls, dr_no_controls, T1_no_controls, age_no_controls, sex_no_controls, study_no_controls, id_no_controls

  elif study == 'ATLAS+NITRC1+NITRC2+OA+MIMS+BMI+NOVA':
    stiffness_all_healthy = np.concatenate((stiffness_ATLAS, stiffness_NITRC_batch_1, stiffness_NITRC_batch_2, stiffness_OA,
                                    stiffness_MIMS, stiffness_BMI, stiffness_NOVA), axis=0)
    dr_all_healthy = np.concatenate((dr_ATLAS, dr_NITRC_batch_1, dr_NITRC_batch_2, dr_OA, dr_MIMS, dr_BMI, dr_NOVA), axis=0)
    T1_all_healthy = np.concatenate((T1_ATLAS, T1_NITRC_batch_1, T1_NITRC_batch_2, T1_OA, T1_MIMS, T1_BMI, T1_NOVA), axis=0)
    age_all_healthy = np.concatenate((age_ATLAS, age_NITRC_batch_1, age_NITRC_batch_2, age_OA, age_MIMS, age_BMI, age_NOVA), axis=0)
    sex_all_healthy = np.concatenate((sex_ATLAS, sex_NITRC_batch_1, sex_NITRC_batch_2, sex_OA, sex_MIMS, sex_BMI, sex_NOVA), axis=0)
    study_all_healthy = np.concatenate((study_ATLAS, study_NITRC_batch_1, study_NITRC_batch_2, study_OA, study_MIMS, study_BMI, study_NOVA), axis=0)
    id_all_healthy = np.concatenate((id_ATLAS, id_NITRC_batch_1, id_NITRC_batch_2, id_OA, id_MIMS, id_BMI, id_NOVA), axis=0)

    if flatten:
      stiffness_all_healthy = stiffness_all_healthy.reshape(stiffness_all_healthy.shape[0],
                                            stiffness_all_healthy.shape[1] * stiffness_all_healthy.shape[2] * stiffness_all_healthy.shape[3])

      dr_all_healthy = dr_all_healthy.reshape(dr_all_healthy.shape[0], dr_all_healthy.shape[1] * dr_all_healthy.shape[2] * dr_all_healthy.shape[3])

      T1_all_healthy = T1_all_healthy.reshape(T1_all_healthy.shape[0], T1_all_healthy.shape[1] * T1_all_healthy.shape[2] * T1_all_healthy.shape[3])

    return stiffness_all_healthy, dr_all_healthy, T1_all_healthy, age_all_healthy, sex_all_healthy, study_all_healthy, id_all_healthy

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


def normalize_max_min(arr, default_value, max_nonzero, min_nonzero):

  arr_nonzero = arr[np.nonzero(arr)]

  if max_nonzero is None and min_nonzero is None:
    max_nonzero = np.max(arr_nonzero)
    min_nonzero = np.min(arr_nonzero)

  if default_value == 0:
    arr_pp = np.zeros_like(arr)

  elif default_value == -1:
    arr_pp = np.ones_like(arr) * -1

  else:
    raise ValueError('default_value has to be 0 or -1')

  arr_pp[np.nonzero(arr)] = (arr[np.nonzero(arr)] - min_nonzero) / (max_nonzero - min_nonzero)

  return arr_pp, max_nonzero, min_nonzero


def norm_whole_batch(batch, norm, default_value):

  batch_normed = np.zeros_like(batch)

  for i in range(batch.shape[0]):
    if norm == 'min_max':
      batch_normed[i], _, _ = normalize_max_min(batch[i], default_value, None, None)

    elif norm == 'mean_std':
      batch_normed[i], _, _ = normalize_mean_0_std_1(batch[i], default_value, None, None)

    else:
      raise ValueError('norm has to be min_max or mean_std')

  return batch_normed

def categorical_one_hot_encoding(category, label):

  if category == 'sex':
    num_classes = 2

    if label == 'M' or label == 'm':
      ind_one = 0
    elif label == 'F' or label == 'f':
      ind_one = 1
    else:
      print(label)
      raise ValueError('label has to be M or F')

  elif category == 'study':
    num_classes = 7

    if label == 'BMI':
      ind_one = 0
    elif label == 'MIMS':
      ind_one = 1
    elif label == 'NITRC_batch_1':
      ind_one = 2
    elif label == 'NITRC_batch_2':
      ind_one = 3
    elif label == 'NOVA':
      ind_one = 4
    elif label == 'ATLAS':
      ind_one = 5
    elif label == 'CN':
      ind_one = 6
    else:
      print(label)
      raise ValueError('label has to be BMI, MIMS, NITRC_batch_1, NITRC_batch_2, NOVA, ATLAS or CN')

  else:
    print(category)
    raise ValueError('category not specified as sex or study')

  arr_one_hot = np.zeros(num_classes)
  arr_one_hot[ind_one] = 1

  return arr_one_hot

def get_lds_kernel_window(kernel, ks, sigma):
  assert kernel in ['gaussian', 'triang', 'laplace']
  half_ks = (ks - 1) // 2
  if kernel == 'gaussian':
    base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
    kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
  elif kernel == 'triang':
    kernel_window = triang(ks)
  else:
    laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
    kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
      map(laplace, np.arange(-half_ks, half_ks + 1)))

  return kernel_window

def num2vect(x, bin_range, bin_step, sigma):
    """
    from Peng, Han, et al. "Accurate brain age prediction with lightweight deep neural networks." Medical image analysis 68 (2021): 101871.
    modified to ensure that the vector sums to 1

    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
      print("bin's range should be divisible by bin_step!")
      return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)

    if sigma == 0:
      x = np.array(x)
      i = np.floor((x - bin_start) / bin_step)
      i = i.astype(int)
      return i, bin_centers
    elif sigma > 0:
      if np.isscalar(x):
        v = np.zeros((bin_number,))
        for i in range(bin_number):
          x1 = bin_centers[i] - float(bin_step) / 2
          x2 = bin_centers[i] + float(bin_step) / 2
          cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
          v[i] = cdfs[1] - cdfs[0]
        # Normalize the vector
        v /= np.sum(v)
        return v, bin_centers
      else:
        v = np.zeros((len(x), bin_number))
        for j in range(len(x)):
          for i in range(bin_number):
            x1 = bin_centers[i] - float(bin_step) / 2
            x2 = bin_centers[i] + float(bin_step) / 2
            cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
            v[j, i] = cdfs[1] - cdfs[0]
          # Normalize each vector
          v[j, :] /= np.sum(v[j, :])
        return v, bin_centers


def random_shift(arr, rand_shift_x, rand_shift_y, rand_shift_z):
  # Augmentation 1: voxel shifting (randomly shift T1_first by 0, 1 or 2 voxels along every axis)
  arr_shifted = np.zeros_like(arr)
  shifts = [rand_shift_x, rand_shift_y, rand_shift_z]
  for i in range(3):
    shift = shifts[i]
    arr_shifted = np.roll(arr, shift, axis=i)
  return arr_shifted

def sagittal_mirror(arr, rand_mirror):
  # Augmentation 2: sagittal mirroring (randomly mirror T1_first along the sagittal plane with probability 0.5)
  if rand_mirror == 1:
    arr = np.flip(arr, axis=0)
  return arr

def rotate_scan(arr, rand_rotate_x, rand_rotate_y, rand_rotate_z):
  arr = rotate(arr, rand_rotate_x, axes=(0, 1), reshape=False, order=0)
  arr = rotate(arr, rand_rotate_y, axes=(1, 2), reshape=False, order=0)
  arr = rotate(arr, rand_rotate_z, axes=(0, 2), reshape=False, order=0)
  return arr

def add_gaussian_noise(arr, sigma):
  noise = np.random.normal(0, sigma, arr.shape)
  arr = arr + noise
  return arr


def augment_batches(batch1, batch2, batch3, mode, augmentations):

  assert all(
    batch.shape == batch1.shape for batch in [batch2, batch3]), "Augment error: Arrays do not have the same shape"

  assert augmentations in ['noise+mirror+rotate+shift', 'mirror+rotate+shift', 'rotate+shift', 'shift'], "Augment error: Invalid augmentations"

  #choose sigma for gaussian noise
  sigma = 0.05

  batch1_aug = np.zeros_like(batch1)
  batch2_aug = np.zeros_like(batch2)
  batch3_aug = np.zeros_like(batch3)

  for i in range(len(batch1)):

    if mode=='inplace':
        rand_mirror = np.random.randint(0, 2)

        rand_shift_x = np.random.randint(0, 3)
        rand_shift_y = np.random.randint(0, 3)
        rand_shift_z = np.random.randint(0, 3)

        rand_rotate_x = np.random.randint(-3, 3)
        rand_rotate_y = np.random.randint(-3, 3)
        rand_rotate_z = np.random.randint(-3, 3)

    elif mode=='additional':
      # Generate random values for mirror and shift until at least one of them is non-zero
        while True:
            rand_mirror = np.random.randint(0, 2)

            rand_shift_x = np.random.randint(0, 3)
            rand_shift_y = np.random.randint(0, 3)
            rand_shift_z = np.random.randint(0, 3)

            rand_rotate_x = np.random.randint(-3, 3)
            rand_rotate_y = np.random.randint(-3, 3)
            rand_rotate_z = np.random.randint(-3, 3)

            if rand_mirror != 0 or rand_shift_x != 0 or rand_shift_y != 0 or rand_shift_z != 0 or rand_rotate_x != 0 or rand_rotate_y != 0 or rand_rotate_z != 0:
                break  # At least one value is non-zero, break the loop

    else:
        raise ValueError('mode has to be inplace or additional')

    if augmentations=='noise+mirror+rotate+shift':

        batch1_aug[i] = random_shift(rotate_scan(sagittal_mirror(add_gaussian_noise(batch1[i], sigma), rand_mirror), rand_rotate_x, rand_rotate_y, rand_rotate_z), rand_shift_x, rand_shift_y, rand_shift_z)
        batch2_aug[i] = random_shift(rotate_scan(sagittal_mirror(add_gaussian_noise(batch2[i], sigma), rand_mirror), rand_rotate_x, rand_rotate_y, rand_rotate_z), rand_shift_x, rand_shift_y, rand_shift_z)
        batch3_aug[i] = random_shift(rotate_scan(sagittal_mirror(add_gaussian_noise(batch3[i], sigma), rand_mirror), rand_rotate_x, rand_rotate_y, rand_rotate_z), rand_shift_x, rand_shift_y, rand_shift_z)

    elif augmentations=='mirror+rotate+shift':

        batch1_aug[i] = random_shift(rotate_scan(sagittal_mirror(batch1[i], rand_mirror), rand_rotate_x, rand_rotate_y, rand_rotate_z), rand_shift_x, rand_shift_y, rand_shift_z)
        batch2_aug[i] = random_shift(rotate_scan(sagittal_mirror(batch2[i], rand_mirror), rand_rotate_x, rand_rotate_y, rand_rotate_z), rand_shift_x, rand_shift_y, rand_shift_z)
        batch3_aug[i] = random_shift(rotate_scan(sagittal_mirror(batch3[i], rand_mirror), rand_rotate_x, rand_rotate_y, rand_rotate_z), rand_shift_x, rand_shift_y, rand_shift_z)

    elif augmentations=='rotate+shift':

        batch1_aug[i] = random_shift(rotate_scan(batch1[i], rand_rotate_x, rand_rotate_y, rand_rotate_z), rand_shift_x, rand_shift_y, rand_shift_z)
        batch2_aug[i] = random_shift(rotate_scan(batch2[i], rand_rotate_x, rand_rotate_y, rand_rotate_z), rand_shift_x, rand_shift_y, rand_shift_z)
        batch3_aug[i] = random_shift(rotate_scan(batch3[i], rand_rotate_x, rand_rotate_y, rand_rotate_z), rand_shift_x, rand_shift_y, rand_shift_z)

    elif augmentations=='shift':

        batch1_aug[i] = random_shift(batch1[i], rand_shift_x, rand_shift_y, rand_shift_z)
        batch2_aug[i] = random_shift(batch2[i], rand_shift_x, rand_shift_y, rand_shift_z)
        batch3_aug[i] = random_shift(batch3[i], rand_shift_x, rand_shift_y, rand_shift_z)

  return batch1_aug, batch2_aug, batch3_aug

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
