import torch
import gpytorch

from models.vanilla_GP import ExactGPModel
from data.datasets import MREDataset
from run_GP import train_GP

def SelectGPModel(x_train, age_train, kernel, location):

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x_train, age_train, likelihood, kernel).double()

    if location == 'cluster':
        model = model.cuda()
        likelihood = likelihood.cuda()

    return model, likelihood

def prepare_datasets(config, device, fold):

    if device == 'cuda':
        location = 'cluster'
    elif device == 'cpu':
        location = 'local'

    # TRAIN DATASET
    train_dataset = MREDataset(modality=config['model']['obs'], train=True, soft_sigma=None, lds=False,
                               noise_std=config['dataset']['noise_std'], location=location, flatten=True, fold=fold)

    if config['dataset']['norm_level'] == 'image_level':
        train_dataset.norm_on_image_level()
    elif config['dataset']['norm_level'] == 'dataset_level':
        train_dataset.norm()
    else:
        raise ValueError("Norm level not recognized")

    # TEST DATASET
    test_dataset = MREDataset(modality=config['model']['obs'], train=False, soft_sigma=None, lds=False, noise_std=None,
                              location=location, flatten=True, fold=fold)

    if config['dataset']['norm_level'] == 'image_level':
        test_dataset.norm_on_image_level()
    elif config['dataset']['norm_level'] == 'dataset_level':
        test_dataset.norm()
    else:
        raise ValueError("Norm level not recognized")

    return [train_dataset, test_dataset]

class GP_trainer:
    def __init__(self, config):

        if torch.cuda.is_available():
            print('on GPU')
            self.device = 'cuda'
            self.location = 'cluster'
        else:
            print('not on GPU')
            self.device = 'cpu'
            self.location = 'local'

        self.kernel = config['model']['kernel']
        self.obs = config['model']['obs']
        self.pca_k = config['model']['pca_k']

        self.biasCorr = config['train']['bias_correction']

        self.nb_epochs = config['train']['nb_epochs']
        self.lr = config['train']['lr']

    def train(self, datasets, save_model, model_path):

        [dataset_train, dataset_test] = datasets

        if self.obs == 'dr+stiffness':
            x_train = torch.cat((torch.from_numpy(dataset_train.x_dr), torch.from_numpy(dataset_train.x_stiff)), dim=-1).double()
            x_test = torch.cat((torch.from_numpy(dataset_test.x_dr), torch.from_numpy(dataset_test.x_stiff)), dim=-1).double()

        else:
            x_train = torch.from_numpy(dataset_train.x).double()
            x_test = torch.from_numpy(dataset_test.x).double()

        age_train = torch.from_numpy(dataset_train.y).double()
        age_test = torch.from_numpy(dataset_test.y).double()

        if self.location == 'cluster':
            x_train = x_train.cuda()
            x_test = x_test.cuda()
            age_train = age_train.cuda()
            age_test = age_test.cuda()

        (U, S, V) = torch.pca_lowrank(x_train, q=self.pca_k, niter=2)

        x_train = torch.matmul(x_train, V[:, :self.pca_k])
        x_test = torch.matmul(x_test, V[:, :self.pca_k])

        model, likelihood = SelectGPModel(x_train=x_train,
                                          age_train=age_train,
                                          kernel=self.kernel,
                                          location=self.location)

        MAE_train, MAE_test = train_GP(location=self.location,
                                        biasCorr=self.biasCorr,
                                        model=model,
                                        likelihood=likelihood,
                                        epochs=self.nb_epochs,
                                        lr=self.lr,
                                        x_train=x_train,
                                        y_train=age_train,
                                        x_test=x_test,
                                        y_test=age_test,
                                        save_model=save_model,
                                        model_path=model_path)

        return MAE_train, MAE_test