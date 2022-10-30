# main_patch_match_train.py
from pytorch_lightning.cli import LightningCLI
from models import patch_matcher
from utils import pairdatamodule

cli = LightningCLI(
        model_class=patch_matcher.PML,
        datamodule_class=pairdatamodule.PairDataModule,
        save_config_overwrite=True,
    )