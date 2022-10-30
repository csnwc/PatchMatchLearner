# main_patch_match_predict.py
from pytorch_lightning.cli import LightningCLI
from dev import patch_match_mnist_dev
from utils import pairdatamodule

cli = LightningCLI(
        model_class=patch_match_mnist_dev.PML_pred,
        datamodule_class=pairdatamodule.PairDataModule,
        save_config_overwrite=True,
    )