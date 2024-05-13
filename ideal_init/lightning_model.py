"""
Pytorch lightning models using IDEAL initialization.

Author
------
Nicolas Rojas
"""
from time import time
from numpy import ndarray
from pandas import read_csv, concat
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch import linalg
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, R2Score
from lightning import LightningModule
from .initialization import init_weights_regression, init_weights_classification


class NNClassifier(LightningModule):
    def __init__(self, X_train: ndarray, y_train: ndarray, X_val: ndarray, y_val: ndarray, X_test: ndarray, y_test: ndarray, initialize: bool = False, hidden_sizes: tuple[int] = None, learning_rate: float = 1e-3, batch_size: int = 64, num_workers: int = 4):
        super().__init__()

        # Set our init args as class attributes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate

        # Normalize data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # Transform numpy matrices to torch tensors
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train)

        self.in_dims = X_train.shape[1]
        self.n_classes = len(torch.unique(y_train))
        binary = self.n_classes == 2
        # Define PyTorch model
        if binary:
            self.metric = Accuracy(task="binary")
            self.loss_fn = F.binary_cross_entropy_with_logits
            self.out_activation = nn.Sigmoid()
            out_shape = 1
        else:
            self.metric = Accuracy(task="multiclass", num_classes=self.n_classes)
            self.loss_fn = F.cross_entropy
            self.out_activation = nn.Softmax(dim=1)
            out_shape = self.n_classes

        self.init_time = time()
        # Check if model is or is not multilayer
        if hidden_sizes is None:
            self.model = nn.Linear(self.in_dims, out_shape)
        else:
            last_shape = self.in_dims
            self.model = nn.Sequential()
            for hidden_size in hidden_sizes:
                self.model.append(nn.Linear(last_shape, hidden_size))
                self.model.append(nn.ReLU())
                last_shape = hidden_size
            self.model.append(nn.Linear(last_shape, out_shape))

        # Initialize model weights if needed
        if initialize:
            self.model = init_weights_classification(self.model, X_train, y_train, weights_method="mean", bias_method="mean")
        self.init_time = time() - self.init_time

        # Create datasets
        if binary:
            y_train = y_train.float().unsqueeze(dim=1)
            y_val = torch.from_numpy(y_val).float().unsqueeze(dim=1)
            y_test = torch.from_numpy(y_test).float().unsqueeze(dim=1)
        else:
            y_train = y_train.long()
            y_val = torch.from_numpy(y_val).long()
            y_test = torch.from_numpy(y_test).long()
        X_val = torch.from_numpy(scaler.transform(X_val)).float()
        X_test = torch.from_numpy(scaler.transform(X_test)).float()

        self.train_data = TensorDataset(X_train, y_train)
        self.val_data = TensorDataset(X_val, y_val)
        self.test_data = TensorDataset(X_test, y_test)

    def forward(self, x):
        logits = self.model(x)
        probas = self.out_activation(logits)
        return probas

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        metric = self.metric(logits, y)

        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train_metric", metric, prog_bar=False, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        metric = self.metric(logits, y)

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("val_metric", metric, prog_bar=True, on_step=True, on_epoch=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        metric = self.metric(logits, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_metric", metric, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        probas = self(x)
        return probas

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)


class NNRegressor(LightningModule):
    def __init__(self, X_train: ndarray, y_train: ndarray, X_val: ndarray, y_val: ndarray, X_test: ndarray, y_test: ndarray, initialize: bool = False, hidden_sizes: tuple[int] = None, learning_rate: float = 1e-3, batch_size: int = 64, num_workers: int = 4):
        super().__init__()

        # Set our init args as class attributes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate

        # Normalize data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # Transform numpy matrices to torch tensors
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()

        self.in_dims = X_train.shape[1]

        # Define PyTorch model
        self.init_time = time()
        if hidden_sizes is None:
            self.model = nn.Linear(self.in_dims, 1)
        else:
            last_shape = self.in_dims
            self.model = nn.Sequential()
            for hidden_size in hidden_sizes:
                self.model.append(nn.Linear(last_shape, hidden_size))
                self.model.append(nn.ReLU())
                last_shape = hidden_size
            self.model.append(nn.Linear(last_shape, 1))

        self.metric = R2Score()
        self.loss_fn = F.mse_loss

        # Initialize model weights if needed
        if initialize:
            self.model = init_weights_regression(self.model, X_train, y_train)
        self.init_time = time() - self.init_time

        # Create datasets
        y_train = y_train.unsqueeze(dim=1)
        y_val = torch.from_numpy(y_val).float().unsqueeze(dim=1)
        y_test = torch.from_numpy(y_test).float().unsqueeze(dim=1)
        X_val = torch.from_numpy(scaler.transform(X_val)).float()
        X_test = torch.from_numpy(scaler.transform(X_test)).float()
        self.train_data = TensorDataset(X_train, y_train)
        self.val_data = TensorDataset(X_val, y_val)
        self.test_data = TensorDataset(X_test, y_test)

    def forward(self, x):
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        metric = self.metric(logits, y)

        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train_metric", metric, prog_bar=False, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        metric = self.metric(logits, y)

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("val_metric", metric, prog_bar=True, on_step=True, on_epoch=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        metric = self.metric(logits, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_metric", metric, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        probas = self(x)
        return probas

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)


class CNNClassifier(LightningModule):
    def __init__(self, X_train: ndarray, y_train: ndarray, X_val: ndarray, y_val: ndarray, X_test: ndarray, y_test: ndarray, initialize: bool = False, learning_rate: float = 1e-3, batch_size: int = 64, num_workers: int = 4):
        super().__init__()

        # Set our init args as class attributes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate

        self.in_dims = X_train.shape[1]
        self.n_classes = len(torch.unique(y_train))
        binary = self.n_classes == 2
        # Define PyTorch model
        if binary:
            self.metric = Accuracy(task="binary")
            self.loss_fn = F.binary_cross_entropy_with_logits
            self.out_activation = nn.Sigmoid()
            out_shape = 1
        else:
            self.metric = Accuracy(task="multiclass", num_classes=self.n_classes)
            self.loss_fn = F.cross_entropy
            self.out_activation = nn.Softmax(dim=1)
            out_shape = self.n_classes

        self.init_time = time()
        self.model = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2880, out_shape),
        )

        X_train /= 255.0
        X_val /= 255.0
        X_test /= 255.0
        # Initialize model weights if needed
        if initialize:
            self.model = init_weights_classification(self.model, X_train, y_train, weights_method="mean", bias_method="mean")
        self.init_time = time() - self.init_time

        # Create datasets
        self.train_data = TensorDataset(X_train, y_train)
        self.val_data = TensorDataset(X_val, y_val)
        self.test_data = TensorDataset(X_test, y_test)

    def forward(self, x):
        logits = self.model(x)
        probas = self.out_activation(logits)
        return probas

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        metric = self.metric(logits, y)

        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train_metric", metric, prog_bar=False, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        metric = self.metric(logits, y)

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("val_metric", metric, prog_bar=True, on_step=True, on_epoch=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        metric = self.metric(logits, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_metric", metric, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        probas = self(x)
        return probas

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)


class extract_tensor(nn.Module):
    def forward(self,x):
        tensor, _ = x
        return tensor[:, -1, :]


class RNNClassifier(LightningModule):
    def __init__(self, X_train: ndarray, y_train: ndarray, X_val: ndarray, y_val: ndarray, X_test: ndarray, y_test: ndarray, initialize: bool = False, learning_rate: float = 1e-3, batch_size: int = 64, num_workers: int = 4):
        super().__init__()

        # Set our init args as class attributes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate

        self.in_dims = X_train.shape[2]
        self.n_classes = len(torch.unique(y_train))
        binary = self.n_classes == 2
        # Define PyTorch model
        if binary:
            self.metric = Accuracy(task="binary")
            self.loss_fn = F.binary_cross_entropy_with_logits
            self.out_activation = nn.Sigmoid()
            out_shape = 1
        else:
            self.metric = Accuracy(task="multiclass", num_classes=self.n_classes)
            self.loss_fn = F.cross_entropy
            self.out_activation = nn.Softmax(dim=1)
            out_shape = self.n_classes

        self.init_time = time()
        self.model = nn.Sequential(
            nn.RNN(self.in_dims, 256, num_layers=2, batch_first=True),
            extract_tensor(),
            nn.Linear(256, out_shape)
        )

        # Initialize model weights if needed
        if initialize:
            self.model = init_weights_classification(self.model, X_train, y_train, weights_method="mean", bias_method="mean")
        self.init_time = time() - self.init_time

        self.train_data = TensorDataset(X_train, y_train)
        self.val_data = TensorDataset(X_val, y_val)
        self.test_data = TensorDataset(X_test, y_test)

    def forward(self, x):
        logits = self.model(x)
        probas = self.out_activation(logits)
        return probas

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        metric = self.metric(logits, y)

        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=False)
        self.log("train_metric", metric, prog_bar=False, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        metric = self.metric(logits, y)

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("val_metric", metric, prog_bar=True, on_step=True, on_epoch=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        metric = self.metric(logits, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_metric", metric, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        probas = self(x)
        return probas

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)


# Get logs from both models
def merge_logs(init_model_logs: str, no_init_model_logs: str):
    init_logs = read_csv(init_model_logs, usecols=["step", "val_metric"]).dropna(axis=0)
    init_logs["method"] = "IDEAL"
    no_init_logs = read_csv(no_init_model_logs, usecols=["step", "val_metric"]).dropna(axis=0)
    no_init_logs["method"] = "He"
    full_logs = concat([init_logs, no_init_logs], ignore_index=True)
    return full_logs
