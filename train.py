import os

from pyannote.database import FileFinder, get_protocol
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from data import SegmentationAndSCDData
from model import SSCDModel

os.environ["PYANNOTE_DATABASE_CONFIG"] = "database.yml"

BATCH_SIZE = 128
DO_SCD = False

protocol = get_protocol(
    "AMI.SpeakerDiarization.mini", preprocessors={"audio": FileFinder()}
)
# num_workers must be 1 if DO_SCD is set due to a bug in the data loading process.
# if DO_SCD == True and num_workers != 1 then batch sizes will not be BATCH_SIZE.
dm = SegmentationAndSCDData(protocol, batch_size=BATCH_SIZE, scd=DO_SCD, num_workers=1)
dm.prepare_data()
dm.setup(stage="fit")
model = SSCDModel(batch_size=BATCH_SIZE, scd=DO_SCD, use_transformer=False)

# Quickly share some parameters. Better method is to use lightning CLI and
# argument linking: https://pytorch-lightning.readthedocs.io/en/1.6.2/common/lightning_cli.html#argument-linking
dm.num_frames = model.num_frames
dm.num_classes = model.num_classes

wandb_logger = WandbLogger(log_model="all")

# trainer = Trainer()
trainer = Trainer(accelerator="gpu", logger=wandb_logger)
trainer.fit(model, datamodule=dm)
