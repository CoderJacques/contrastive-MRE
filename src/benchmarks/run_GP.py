import torch
import gpytorch
import numpy as np
from sklearn import linear_model


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if (validation_loss + self.min_delta) < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_GP(location, biasCorr, model, likelihood, epochs, lr, x_train, y_train, x_test, y_test, save_model,
             model_path):

    if location == 'cluster':
        path_prefix = '/rds/user/jnt27/hpc-work/BrainAgeMRE/GPs_trained_models/'
    else:
        path_prefix = '/Users/jakobtraeuble/PycharmProjects/BrainAgeMRE/trained_models/'

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    if location == 'cluster':
        mll = mll.cuda()

    for i in range(epochs):

        ##TRAIN
        model.train()
        likelihood.train()

        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()

        optimizer.step()

        ##EVAL
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            age_pred_train = np.array(likelihood(model(x_train)).mean.cpu())
            age_pred_test = np.array(likelihood(model(x_test)).mean.cpu())

        if biasCorr:
            biasCorrModel = linear_model.LinearRegression()
            biasCorrModel.fit(np.array(y_train.cpu()).reshape(-1, 1), age_pred_train.reshape(-1, 1))

            age_pred_train = torch.from_numpy(((age_pred_train - biasCorrModel.intercept_) / biasCorrModel.coef_).squeeze())
            age_pred_test = torch.from_numpy(((age_pred_test - biasCorrModel.intercept_) / biasCorrModel.coef_).squeeze())

        #calculate mean absolute error between np arrays y_train and age_pred_train
        MAE_train = np.abs(y_train.cpu().numpy() - age_pred_train).mean()
        MAE_test = np.abs(y_test.cpu().numpy() - age_pred_test).mean()

    if save_model:
        print('Saving model...')
        torch.save(model.state_dict(), path_prefix + model_path)

    return MAE_train, MAE_test

def evaluate_GP(model, likelihood, x_test, y_test):

    ##EVAL
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        age_pred_test = np.array(likelihood(model(x_test)).mean.cpu())

    #calculate mean absolute error between np arrays y_train and age_pred_train
    MAE_test = np.abs(y_test.cpu().numpy() - age_pred_test).mean()

    return MAE_test