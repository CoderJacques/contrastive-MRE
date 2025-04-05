import torch
from tqdm import tqdm
import numpy as np
import os
import random
import wandb


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)


class EarlyStopper:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def batch_iteration(obs, device, model, x, y):

    if obs == "stiffness":
        inputs = x.to(device)
        inputs = torch.unsqueeze(inputs, 1)

    elif obs == "dr":
        inputs = x.to(device)
        inputs = torch.unsqueeze(inputs, 1)

    elif obs == "T1":
        inputs = x.to(device)
        inputs = torch.unsqueeze(inputs, 1)

    elif obs == "dr+stiffness":
        (x_dr, x_stiff) = x
        dr_map = x_dr.to(device)
        dr_map = torch.unsqueeze(dr_map, 1)
        stiffness_map = x_stiff.to(device)
        stiffness_map = torch.unsqueeze(stiffness_map, 1)
        inputs = torch.cat((dr_map, stiffness_map), dim=1).to(device)

    else:
        raise ValueError("No observation specified")

    inputs = inputs.to(device)
    labels = y.to(device)

    y, features = model(inputs)

    return y, features, labels


def train_loop(nb_epochs, model, optimizer, train_loader, val_loader, device, obs, weighted, loss, save_model,
               model_path, early_stopping):

    if device == 'cpu':
        path_prefix = '/Users/jakobtraeuble/PycharmProjects/BrainAgeMRE/trained_models/'
    elif device == 'cuda':
        path_prefix = '/rds/user/jnt27/hpc-work/BrainAgeMRE/BrainAgeMRE_trained_models/'

    if early_stopping:
        early_stopper = EarlyStopper(patience=5)

    val_loss_min = np.inf

    for epoch in range(nb_epochs):

        total_train_loss = 0.0
        total_train_samples = 0

        train_labels = []
        train_predictions = []

        ## Training step
        model.train()
        nb_batch = len(train_loader)

        if device == 'cpu':
            pbar = tqdm(total=nb_batch, desc="Training")

        for x, y, sex, site, weights in train_loader:

            optimizer.zero_grad()

            if device == 'cpu':
                pbar.update()

            y, features, labels = batch_iteration(obs=obs, device=device, model=model, x=x, y=y)

            batch_size = labels.size(0)

            if y.shape[-1] > 1:  # soft classification case
                bin_centers = 0.5 + float(1) / 2 + 1 * np.arange(100)
                y_int = np.exp(torch.squeeze(y).detach().cpu().numpy()) @ bin_centers
                labels_int = np.argmax(labels.detach().cpu().numpy(), axis=1) + 1

                train_labels.append(labels_int.tolist())
                train_predictions.append(y_int.tolist())
            else:
                train_labels.append(torch.squeeze(labels).detach().cpu().numpy().tolist())
                train_predictions.append(torch.squeeze(y).detach().cpu().numpy().tolist())

            if weighted:
                weights = weights.to(device).squeeze(1)
            else:
                weights = None

            train_loss = loss(torch.squeeze(y), labels, weights)

            train_loss.backward()

            optimizer.step()

            # Accumulate the weighted loss
            total_train_loss += train_loss.detach().cpu().numpy() * batch_size

            # Accumulate the total number of samples
            total_train_samples += batch_size

        if device == 'cpu':
            pbar.close()

        ## Validation step
        nb_batch = len(val_loader)
        if device == 'cpu':
            pbar = tqdm(total=nb_batch, desc="Validation")

        train_loss_epoch = total_train_loss / total_train_samples

        #log the training loss in wandb
        wandb.log({"train/loss": train_loss_epoch, "lr": optimizer.param_groups[0]['lr'], "epoch": epoch})

        total_val_loss = 0.0
        total_val_samples = 0

        val_labels = []
        val_predictions = []

        with torch.no_grad():
            model.eval()
            for x, y, sex, site, weights in val_loader:

                if device == 'cpu':
                    pbar.update()

                y, features, labels = batch_iteration(obs=obs, device=device, model=model, x=x, y=y)

                batch_size = labels.size(0)

                val_loss = loss(torch.squeeze(y), labels)

                # Accumulate the weighted loss
                total_val_loss += val_loss.detach().cpu().numpy() * batch_size

                # Accumulate the total number of samples
                total_val_samples += batch_size

                if y.shape[-1] > 1:  # soft classification case
                    bin_centers = 0.5 + float(1) / 2 + 1 * np.arange(100)
                    y_int = np.exp(torch.squeeze(y).detach().cpu().numpy()) @ bin_centers
                    labels_int = np.argmax(labels.detach().cpu().numpy(), axis=1) + 1

                    val_labels.append(labels_int.tolist())
                    val_predictions.append(y_int.tolist())

                else:
                    val_labels.append(torch.squeeze(labels).detach().cpu().numpy().tolist())
                    val_predictions.append(torch.squeeze(y).detach().cpu().numpy().tolist())

            val_loss_epoch = total_val_loss / total_val_samples
            wandb.log({"val/loss": val_loss_epoch, "lr": optimizer.param_groups[0]['lr'], "epoch": epoch})

            # calculate mae_train
            train_labels = [item for sublist in train_labels for item in sublist]
            train_predictions = [item for sublist in train_predictions for item in sublist]
            mae_train = np.mean(np.abs(np.array(train_labels) - np.array(train_predictions)))

            # calculate mae_test
            val_labels = [item for sublist in val_labels for item in sublist]
            val_predictions = [item for sublist in val_predictions for item in sublist]
            mae_val = np.mean(np.abs(np.array(val_labels) - np.array(val_predictions)))

            wandb.log({"train/mae": mae_train, "epoch": epoch})
            wandb.log({"val/mae": mae_val, "epoch": epoch})

        if early_stopping:
            if val_loss_epoch < val_loss_min:
                val_loss_min = val_loss_epoch

                #bin_centers = 0.5 + float(1) / 2 + 1 * np.arange(100)

                #plot_label_vs_prediction(true_label=labels.detach().cpu().numpy()[0],
                #                         predicted_label=np.exp(torch.squeeze(y[0]).detach().cpu().numpy()),
                #                         bin_centers=bin_centers,
                #                         prefix='Fold_'+str(fold)+'_Epoch_'+str(epoch+1))

                if save_model:
                    print('Saving model...')
                    torch.save(model.state_dict(), path_prefix + model_path)

        if save_model:
            print('Saving model...')
            torch.save(model.state_dict(), path_prefix + model_path)

        if early_stopping:
            if early_stopper.early_stop(val_loss):
                break

        if device == 'cpu':
            pbar.close()

        print("Epoch [{current_epoch}/{nb_epochs}] Training loss = {training_loss}\t Validation loss = {val_loss}\t".format(
            current_epoch=epoch+1, nb_epochs=nb_epochs, training_loss=train_loss_epoch, val_loss=val_loss_epoch))

    return mae_train, mae_val

def evaluate_loop(model, test_loader, device, obs):

        val_labels = []
        val_predictions = []

        with torch.no_grad():
            model.eval()
            for x, y, sex, site, weights in test_loader:

                y, features, labels = batch_iteration(obs=obs, device=device, model=model, x=x, y=y)

                if y.shape[-1] > 1:  # soft classification case
                    bin_centers = 0.5 + float(1) / 2 + 1 * np.arange(100)
                    y_int = np.exp(torch.squeeze(y).detach().cpu().numpy()) @ bin_centers
                    labels_int = np.argmax(labels.detach().cpu().numpy(), axis=1) + 1

                    val_labels.append(labels_int.tolist())
                    val_predictions.append(y_int.tolist())

                else:
                    val_labels.append(torch.squeeze(labels).detach().cpu().numpy().tolist())
                    val_predictions.append(torch.squeeze(y).detach().cpu().numpy().tolist())

            # calculate mae_test
            val_labels = [item for sublist in val_labels for item in sublist]
            val_predictions = [item for sublist in val_predictions for item in sublist]
            mae_val = np.mean(np.abs(np.array(val_labels) - np.array(val_predictions)))

        return mae_val