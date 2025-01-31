import torch
print("Current default dtype:", torch.get_default_dtype())
torch.set_default_dtype(torch.float32)  # Set default dtype to float32
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np
import random
import copy

from scipy import interpolate

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class ControlledDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ControlledDropout, self).__init__()
        self.p = p
        self.dropout_enabled = False  # Initially, set to False

    def forward(self, x):
        if self.training:
            # Apply dropout in training mode as usual
            return F.dropout(x, self.p, self.training)
        elif self.dropout_enabled:
            # Apply dropout in eval mode if enabled
            return F.dropout(x, 0.1, True)
        else:
            # No dropout if disabled in eval mode
            return x

    def enable_dropout(self):
        self.dropout_enabled = True

    def disable_dropout(self):
        self.dropout_enabled = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Dataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        x = torch.Tensor(self.x[index])
        y = torch.Tensor(self.y[index])
        return (x, y)

    def __len__(self):
        count = self.x.shape[0]
        return count


class MCCQR_MLP_Regressor(nn.Module):
    def __init__(self, input_dim, hidden_dim1, quantile_fits=None, dropout_rate=0.2, device='cpu'):
        super(MCCQR_MLP_Regressor, self).__init__()
        self.device = torch.device(device)
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.n_outputs = len(quantile_fits) if quantile_fits is not None else 101
        self.dropout_rate = dropout_rate
        self.quantile_fits = torch.tensor(quantile_fits if quantile_fits is not None else np.arange(0.01, 1.01, 0.01),
                                          dtype=torch.float32, device=self.device)

        # Define layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = ControlledDropout(dropout_rate)
        self.fc2 = nn.Linear(self.hidden_dim1, self.n_outputs)

        self.calibration_constant = 0  # Default calibration constant

    def forward(self, x, training=False):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

    def calculate_calibration_constant(self, X_cal, y_cal, alpha=0.1):
        """ Calculate and store the calibration constant using a calibration dataset """
        self.eval()  # Ensure the model is in evaluation mode
        preds = self.predict(X_cal, n_draws=1000, apply_calibration=False)  # Get predictions for calibration data

        lower_preds = preds['0.010_aleatory_epistemic']
        upper_preds = preds['1.000_aleatory_epistemic']

        # Calculation of non-conformity scores where scores are zero if y_cal is within the predicted interval
        scores = np.where((y_cal < lower_preds) | (y_cal > upper_preds),
                          np.maximum(np.abs(y_cal - upper_preds), np.abs(lower_preds - y_cal)),
                          0)

        self.calibration_constant = np.quantile(scores, 1 - alpha)  # Set the calibration constant

    def tilted_loss(self, y, f):
        e = (y - f)
        losses = torch.where(e > 0, self.quantile_fits * e, (self.quantile_fits - 1) * e)
        return torch.mean(losses)

    # @no_grad()
    def predict(self, X, n_draws=1000, apply_calibration=True):
        print('predicting')
        self.eval()  # Set the model to evaluation mode
        val_dict = {}
        quantiles = np.asarray([q / 100 for q in range(1, 101)])  # Assuming 101 quantiles from 1% to 100%

        uni_rand = torch.rand(n_draws, device=self.device).numpy()  # Ensure it's on the same device as the model

        with torch.no_grad():  # Disable gradient calculation for inference
            for id in ['_noEpistemic', '_epistemic']:
                q_preds_aleatory = []
                q_preds_noAleatory = []
                if id == '_epistemic':
                    self.dropout1.enable_dropout()
                    for i in range(n_draws):
                        y_pred = torch.sigmoid(self((torch.tensor(X)).float()).squeeze())* self.y_span + self.y_lower
                        y_pred = y_pred.cpu().numpy() # Convert to numpy array for SciPy
                        interp_cdf = interpolate.interp1d(self.quantile_fits.cpu().numpy(),
                                                          y_pred,
                                                          axis=1,
                                                          fill_value='extrapolate')

                        q_preds_aleatory.append(interp_cdf(uni_rand[i]))
                        q_preds_noAleatory.append(interp_cdf(0.5))

                        if i % 100 == 0:
                            print(f'Drawing with Epistemic Uncertainty {i + 1}/{n_draws} {id[1:]}')

                    # build output
                    val_dict = self.fill_val_dict(quantiles, val_dict, np.asarray(q_preds_aleatory), '_aleatory' + id)
                    val_dict = self.fill_val_dict(quantiles, val_dict, np.asarray(q_preds_noAleatory), '_noAleatory' + id)

                elif id == '_noEpistemic':
                    self.dropout1.disable_dropout()
                    y_pred = torch.sigmoid(self((torch.tensor(X)).float()).squeeze())* self.y_span + self.y_lower
                    y_pred = y_pred.cpu().numpy()
                    interp_cdf = interpolate.interp1d(self.quantile_fits.cpu().numpy(),
                                                      y_pred,
                                                      axis=1,
                                                      fill_value='extrapolate')

                    q_preds_aleatory.append(interp_cdf(uni_rand))
                    q_preds_noAleatory.append(interp_cdf(0.5))
                    print('No Epistemic Uncertainty.')

                    # Process and fill the value dictionary for each condition
                    self.fill_val_dict(quantiles, val_dict, np.asarray(np.squeeze(q_preds_aleatory)).transpose(), '_aleatory' + id)
                    self.fill_val_dict(quantiles, val_dict, np.array(q_preds_noAleatory), '_noAleatory' + id, do_quants=False)

                if apply_calibration:
                    # Apply the stored calibration constant
                    for key in val_dict.keys():
                        val_dict[key] -= self.calibration_constant  # Adjusting all quantile predictions

        y_pred = val_dict['median_noAleatory_epistemic']

        val_dict["y_pred"] = y_pred
        return val_dict

    def fill_val_dict(self, quantiles, val_dict, q_preds, id_suffix, do_quants=True):
        q_preds = np.atleast_2d(q_preds)

        val_dict["median" + id_suffix] = np.median(q_preds, axis=0)
        val_dict["mean" + id_suffix] = np.mean(q_preds, axis=0)
        val_dict["std" + id_suffix] = np.std(q_preds, axis=0)
        val_dict["mad" + id_suffix] = np.median(np.abs(q_preds - np.median(q_preds, axis=0)), axis=0)
        q_out = np.quantile(a=q_preds, q=quantiles, axis=0)

        if np.all(np.diff(q_out) < 0):
            print('Quantile cross-over!')

        # build output dict
        if do_quants:
            for i, q in enumerate(quantiles):
                val_dict[f"{q:.3f}" + id_suffix] = q_out[i, :]
        return val_dict

    def init_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()  # Reset parameters according to layer's built-in method

            # Convert all parameters to float32
            for param in layer.parameters():
                param.data = param.data.float()

            # Convert all buffers to float32, specifically important for batch norm layers
            if hasattr(layer, 'buffers'):
                for buf in layer.buffers():
                    if buf is not None:
                        buf.data = buf.data.float()

    def fit(self, Xtrain, ytrain, Xval, yval, fold, epochs=500, lr=0.01, weight_decay=1e-4):
        X_val = (torch.tensor(Xval)).float()
        y_val = torch.tensor(yval).float()
        X = (torch.tensor(Xtrain)).float()
        y = torch.tensor(ytrain).float()

        self.y_span = max(y)-min(y)
        self.y_lower = min(y)

        self.init_params()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        # Scheduler: Reduce learning rate when validation loss plateaus
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

        best_epoch = 0
        best_mae_val = np.inf
        best_val_loss = np.inf  # Initialize the best validation loss
        val_improve_epoch = 0
        epochs_wo_improve = 20

        train_loss_list, val_loss_list, epoch_list = [], [], []
        train_mae_list, val_mae_list = [], []

        # Save the best model's state
        best_model_state = self.state_dict()

        for epoch in range(epochs):
            train_dataset = Dataset(X, y)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=int(X.shape[0] / int(X.shape[0]/64)),
                                                       shuffle=True, drop_last=True)
            total_train_loss = 0.0
            total_train_mae = 0.0

            for train_data in train_loader:
                self.train()
                # y_pred = self(train_data[0]).squeeze()
                y_pred = torch.sigmoid((self(train_data[0])).squeeze()) * self.y_span + self.y_lower

                loss = self.tilted_loss(y_pred, train_data[1].reshape(-1, 1))
                total_train_loss += loss.item()

                total_train_mae += mean_absolute_error(train_data[1].reshape(-1, 1).detach().numpy(),
                                    ((y_pred[:, 49] + y_pred[:, 50]) / 2).detach().numpy())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluation
            self.eval()
            with torch.no_grad():
                y_pred_val = torch.sigmoid(self(X_val).squeeze())* self.y_span + self.y_lower
                val_loss = self.tilted_loss(y_pred_val, y_val.reshape(-1, 1)).item()  # Compute validation loss
                mae_val = mean_absolute_error(y_val.detach().numpy(), ((y_pred_val[:, 49]+y_pred_val[:, 50])/2).detach().numpy())

                if (mae_val < best_mae_val):
                    val_improve_epoch = epoch
                    best_mae_val = min(mae_val, best_mae_val)
                    best_val_loss = min(val_loss, best_val_loss)  # Update best validation loss
                    best_epoch = epoch
                    # Consider deepcopy to save the best model state
                    best_model_state = copy.deepcopy(self.state_dict())

                if epoch - val_improve_epoch >= epochs_wo_improve:
                    break

            scheduler.step(val_loss)

            train_loss_list.append(total_train_loss/len(train_loader))
            val_loss_list.append(val_loss)

            train_mae_list.append(total_train_mae/len(train_loader))
            val_mae_list.append(mae_val)

            epoch_list.append(epoch)
            print(f'Epoch: {epoch}, Train Loss: {total_train_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Train MAE: {total_train_mae/len(train_loader):.4f} Validation MAE: {mae_val:.4f},')

        plt.figure(figsize=(10, 6))
        plt.plot(epoch_list, train_loss_list, label='Training Loss', marker=None, color='blue')
        plt.plot(epoch_list, val_loss_list, label='Validation Loss', marker=None, color='red')
        plt.plot(epoch_list, train_mae_list, label='Training MAE', marker=None, color='orange')
        plt.plot(epoch_list, val_mae_list, label='Validation MAE', marker=None, color='purple')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Saving the plot as a PNG file
        plt.savefig('training_validation_loss_'+str(fold)+'.svg')

        print(
            f'No improvement in {epochs_wo_improve} epochs, returning best model at epoch: {best_epoch}, Best MAE: {best_mae_val:.4f}, Best Validation Loss: {best_val_loss:.4f}')

        # Load the best model state
        self.load_state_dict(best_model_state)
        return self
