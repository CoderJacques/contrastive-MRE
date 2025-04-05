import torch

from data.datasets import MREDataset
from run_CNN import train_loop
from losses import weighted_mse_loss, weighted_l1_loss, weighted_focal_mse_loss, weighted_focal_l1_loss, weighted_huber_loss, my_KLDivLoss
from models.resnet import ResNet


def SelectModel(net_arc, mode, input_channels, latent_dim, dropout, device):

    if net_arc == 'ResNet':
        net = ResNet(mode=mode,
                          input_channels=input_channels,
                          latent_dim=latent_dim,
                          dropout=dropout,
                          device=device).to(torch.float32).to(device)

    return net


def prepare_model(config, device):

    model_config = config['model']

    net_arc = model_config['net']
    mode = model_config['mode']
    obs = model_config['obs']
    latent_dim = model_config['latent_dim']
    dropout = model_config['dropout']

    if obs in ['dr', 'stiffness', 'T1']:
        input_channels = 1
    elif obs in ['dr+stiffness', 'T1+dr', 'T1+stiffness']:
        input_channels = 2
    elif obs == 'T1+dr+stiffness':
        input_channels = 3
    else:
        raise ValueError('Invalid obs variable - choose [dr, stiffness, T1, dr+stiffness, T1+dr, T1+stiffness, T1+dr+stiffness]')

    if net_arc == 'ResNet':
        model = ResNet(mode=mode,
                            input_channels=input_channels,
                            latent_dim=latent_dim,
                            dropout=dropout,
                            device=device).to(torch.float32).to(device)

    return model


def prepare_dataloader(config, device, fold):

    if device == 'cuda':
        location = 'cluster'
    elif device == 'cpu':
        location = 'local'

    # TRAIN DATASET
    train_dataset = MREDataset(modality=config['model']['obs'], train=True, soft_sigma=config['dataset']['soft_sigma'],
                               lds=config['dataset']['LDS'], noise_std=config['dataset']['noise_std'],
                               location=location, flatten=False, fold=fold)

    if config['dataset']['norm_level'] == 'image_level':
        print('norm with norm_on_image_level()')
        train_dataset.norm_on_image_level()
    else:
        train_dataset.norm()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True)

    # TEST DATASET
    test_dataset = MREDataset(modality=config['model']['obs'], train=False, soft_sigma=config['dataset']['soft_sigma'],
                              lds=config['dataset']['LDS'], noise_std=None, location=location, flatten=False, fold=fold)

    if config['dataset']['norm_level'] == 'image_level':
        print('norm with norm_on_image_level()')
        test_dataset.norm_on_image_level()
    else:
        test_dataset.norm()

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    return [train_loader, test_loader]


class CNN_trainer:
    def __init__(self, config, device):

        self.train_config = config['train']

        self.device = device

        self.nb_epochs = self.train_config['nb_epochs']
        self.lr = self.train_config['lr']
        self.weight_decay = self.train_config['weight_decay']
        self.loss = self.train_config['loss']
        self.biasCorr = self.train_config['bias_correction']

        print('...trainer initialized')
        print('...lr: ', self.lr)
        print('...weight_decay: ', self.weight_decay)

        self.obs = config['model']['obs']
        #self.FDS = config['model']['FDS']
        self.LDS = config['dataset']['LDS']

        self.config = config

        if self.LDS:
            self.weighted = True
        else:
            self.weighted = False

        if self.loss == 'mse':
            self.loss = weighted_mse_loss
        elif self.loss == 'l1':
            self.loss = weighted_l1_loss
        elif self.loss == 'focal_mse':
            self.loss = weighted_focal_mse_loss
        elif self.loss == 'focal_l1':
            self.loss = weighted_focal_l1_loss
        elif self.loss == 'huber':
            self.loss = weighted_huber_loss
        elif self.loss == 'KL':
            self.loss = my_KLDivLoss
        else:
            raise ValueError('No valid loss function specified')

    def train(self, model, dataloader, model_path):

        [loader_train, loader_test] = dataloader

        if model_path == "":
            save_model = False
        else:
            save_model = True

        if self.train_config['optimizer'] == 'adam':
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                              weight_decay=self.weight_decay)
        elif self.train_config['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.lr,
                                             weight_decay=self.weight_decay)

        mae_train, mae_val = train_loop(nb_epochs=self.nb_epochs,
                                                       model=model,
                                                       optimizer=self.optimizer,
                                                       train_loader=loader_train,
                                                       val_loader=loader_test,
                                                       device=self.device,
                                                       obs=self.obs,
                                                       weighted=self.weighted,
                                                       loss=self.loss,
                                                       save_model=save_model,
                                                       model_path=model_path,
                                                       early_stopping=self.train_config['early_stopping'])

        return mae_train, mae_val