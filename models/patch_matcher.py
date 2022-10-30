# patch_matcher.py
from typing import Tuple, List, Dict, Type, Any
from pytorch_lightning import LightningModule
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

class PML(LightningModule):
    '''
    Spatio-temporal Patch Match Learner, aka block matching across a shift pair.

    x = [B,1,2,H,W]
    '''
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        bias: bool = True,
        learning_rate: float = 1e-4,
        optimizer: Type[Optimizer] = Adam,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        **kwargs: Any,):
        super(PML, self).__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        # self.optimizer = self.configure_optimizers()
        
        self.conv1 = nn.Conv3d(1, 8, (2, 3, 3), stride=1)
        self.conv2 = nn.Conv2d(8, 4 , 3, stride=2)
        linear_input_size = int(((((input_dim-2)/2)-1)**2)*4) # 484 for input_dim=28
        self.head = nn.Linear(linear_input_size, output_dim, bias=bias)

    def forward(self, x):

        x = self.conv1(x)
        x = x.squeeze(2) # Take out frames - we've reduced dim2 to 1
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        y_pred = self.head(x.view(x.size(0), -1))
        return torch.squeeze(y_pred)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        x, y = batch[0], batch[1]

        y_hat = self(x)

        loss = F.mse_loss(y_hat, y, reduction="mean") #"sum")

        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = self.linear.weight.abs().sum()
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = self.linear.weight.pow(2).sum()
            loss += self.hparams.l2_strength * l2_reg

        loss /= x.size(0)

        self.log("train_mse_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch[0], batch[1]
        y_hat = self(x)
        return {"val_loss": F.mse_loss(y_hat, y)}

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_mse_loss", val_loss, sync_dist=True)
        return # {"val_loss": val_loss, "log": tensorboard_logs, "progress_bar": progress_bar_metrics}

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        x, y = batch
        y_hat = self(x)
        return {"test_loss": F.mse_loss(y_hat, y)}

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)