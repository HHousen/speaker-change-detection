import math

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pyannote.audio.torchmetrics import (
    OptimalDiarizationErrorRate,
    OptimalDiarizationErrorRateThreshold,
    OptimalFalseAlarmRate,
    OptimalMissedDetectionRate,
    OptimalSpeakerConfusionRate,
)
from pyannote.audio.utils.loss import binary_cross_entropy
from pyannote.audio.utils.permutation import permutate
from torchmetrics import AUROC, MetricCollection

from sincnet import SincNet


class SSCDModel(pl.LightningModule):
    """Segmentation and speaker change detection (SCD) model.
    Pipeline: SincNet > LSTM > Linear > Classifier
    """

    def __init__(
        self,
        sincnet={"stride": 10},
        lstm={
            "hidden_size": 128,
            "num_layers": 4,
            "bidirectional": True,
            "dropout": 0.5,
            "batch_first": True,
        },
        sample_rate: int = 16000,
        batch_size=32,
        duration=5,
        num_classes=4,
        scd=False,
        use_transformer=False,
    ):
        super().__init__()

        self.duration = duration
        self.batch_size = batch_size
        self.num_classes = num_classes
        if scd:
            self.num_classes = 1
        self.scd = scd
        self.use_transformer = use_transformer

        sincnet["sample_rate"] = sample_rate
        self.save_hyperparameters()

        self.sincnet = SincNet(**sincnet)

        if self.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=60, nhead=6)
            self.core = nn.TransformerEncoder(encoder_layer, num_layers=4)
            num_out_features = 60
        else:
            self.core = nn.LSTM(input_size=60, **lstm)
            num_out_features = lstm["hidden_size"] * (2 if lstm["bidirectional"] else 1)

        num_out_features_halved = num_out_features // 2
        self.linear = nn.ModuleList(
            [
                nn.Linear(
                    in_features=num_out_features,
                    out_features=num_out_features_halved,
                    bias=True,
                ),
                nn.Linear(
                    in_features=num_out_features_halved,
                    out_features=num_out_features_halved,
                    bias=True,
                ),
            ]
        )

        self.classifier = nn.Linear(
            in_features=num_out_features_halved, out_features=self.num_classes
        )
        self.activation = nn.Sigmoid()

        self.get_num_frames()

        self.validation_metric = MetricCollection(
            [
                AUROC(
                    self.num_frames if self.scd else self.num_classes,
                    pos_label=1,
                    average="macro",
                    compute_on_step=False,
                ),
                OptimalDiarizationErrorRate(),
                OptimalDiarizationErrorRateThreshold(),
                OptimalSpeakerConfusionRate(),
                OptimalMissedDetectionRate(),
                OptimalFalseAlarmRate(),
            ]
        )

    def get_num_frames(self):
        x = torch.randn(
            (
                self.batch_size,
                1,  # 1 channel
                int(self.hparams.sample_rate * self.duration),
            ),
        )
        with torch.no_grad():
            self.num_frames = self.sincnet(x).shape[-1]
        return self.num_frames

    def forward(self, waveforms):
        """Accepts a waveform of shape (batch, channel, sample) and returns scores
        of shape (batch, frame, classes).
        """

        # waveforms.shape = batch_size, 1, 80000
        outputs = self.sincnet(waveforms)

        if self.use_transformer:
            outputs = rearrange(outputs, "batch feature frame -> frame batch feature")
            outputs = self.core(outputs)
            outputs = rearrange(outputs, "frame batch feature -> batch frame feature")
        else:
            outputs = rearrange(outputs, "batch feature frame -> batch frame feature")
            outputs, _ = self.core(outputs)

        for linear in self.linear:
            outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))

    def loss_func(
        self, prediction, target,
    ):
        # prediction is of shape (batch_size, num_frames, num_classes)
        # target is of shape (batch_size, num_frames, num_speakers)

        return binary_cross_entropy(prediction, target.float())

    def training_step(self, batch, batch_idx):
        target = batch["y"]
        # (batch_size, num_frames, num_speakers)

        waveform = batch["X"]
        # (batch_size, num_channels, num_samples)

        # drop samples that contain too many speakers
        if not self.scd:
            num_speakers: torch.Tensor = torch.sum(torch.any(target, dim=1), dim=1)
            keep: torch.Tensor = num_speakers <= self.num_classes
            target = target[keep]
            waveform = waveform[keep]

            if not keep.any():
                return {"loss": 0.0}

        # forward pass
        prediction = self.forward(waveform)
        _, num_frames, _ = prediction.shape
        # (batch_size, num_frames, num_classes)
        assert (
            num_frames == self.num_frames
        ), "num_frames mismatch, audio length not 5s?"

        if self.scd:
            loss = self.loss_func(prediction, target)
        else:
            # Find optimal permutation for permutation-invariant segmentation loss
            permutated_prediction, _ = permutate(target, prediction)

            loss = self.loss_func(permutated_prediction, target)

        self.log(
            "TrainLoss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )

        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        X, y = batch["X"], batch["y"]
        # X = (batch_size, num_channels, num_samples)
        # y = (batch_size, num_frames, num_classes) or (batch_size, num_frames)

        y_pred = self.forward(X)
        # y_pred = (batch_size, num_frames, num_classes)

        target = y

        if self.scd:
            loss = self.loss_func(y_pred, target)
        else:
            # Find optimal permutation for permutation-invariant segmentation loss
            permutated_prediction, _ = permutate(target, y_pred)

            loss = self.loss_func(permutated_prediction, target)

        self.log(
            "ValLoss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,
        )

        # Pad target with zeros if necessary to reach same num_classes/num_speakers
        if target.shape[-1] != y_pred.shape[-1]:
            pad_func = nn.ConstantPad1d((0, y_pred.shape[-1] - target.shape[-1]), 0)
            target = pad_func(y)

        # target: shape (batch_size, num_frames, num_classes), type binary
        # preds:  shape (batch_size, num_frames, num_classes), type float
        # torchmetrics expects
        # target: shape (batch_size, num_classes, ...), type binary
        # preds:  shape (batch_size, num_classes, ...), type float

        self.validation_metric(
            y_pred.squeeze() if self.scd else torch.transpose(y_pred, 1, 2),
            target.squeeze() if self.scd else torch.transpose(target, 1, 2),
        )

        self.log_dict(
            self.validation_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # 3 for SCD, bs=128, transformer=False
        # 2 for for SCD, bs=64, transformer=True
        # 3 for segmentation, bs=128, transformer=False
        if batch_idx == 3:
            # Visualize first 9 validation samples of first batch
            # From https://github.com/pyannote/pyannote-audio/blob/3147e2bfe9a7af388d0c01f3bba3d0578ba60c67/pyannote/audio/tasks/segmentation/mixins.py#L422
            X = X.cpu().numpy()
            y = y.float().cpu().numpy()
            y_pred = y_pred.cpu().numpy()

            # prepare 3 x 3 grid (or smaller if batch size is smaller)
            num_samples = min(self.batch_size, 9)
            nrows = math.ceil(math.sqrt(num_samples))
            ncols = math.ceil(num_samples / nrows)
            fig, axes = plt.subplots(
                nrows=2 * nrows, ncols=ncols, figsize=(8, 5), squeeze=False
            )

            # reshape target so that there is one line per class when plotting it
            y[y == 0] = np.NaN
            if len(y.shape) == 2:
                y = y[:, :, np.newaxis]
            y *= np.arange(y.shape[2])

            # plot each sample
            for sample_idx in range(num_samples):

                # find where in the grid it should be plotted
                row_idx = sample_idx // nrows
                col_idx = sample_idx % ncols

                # plot target
                ax_ref = axes[row_idx * 2 + 0, col_idx]
                sample_y = y[sample_idx]
                ax_ref.plot(sample_y)
                ax_ref.set_xlim(0, len(sample_y))
                ax_ref.set_ylim(-1, sample_y.shape[1])
                ax_ref.get_xaxis().set_visible(False)
                ax_ref.get_yaxis().set_visible(False)

                # plot predictions
                ax_hyp = axes[row_idx * 2 + 1, col_idx]
                sample_y_pred = y_pred[sample_idx]

                ax_hyp.plot(sample_y_pred)
                ax_hyp.set_ylim(-0.1, 1.1)
                ax_hyp.set_xlim(0, len(sample_y))
                ax_hyp.get_xaxis().set_visible(False)

            plt.tight_layout()

            # tensorboard
            # self.logger.experiment.add_figure(
            #     "ValSamples", fig, self.current_epoch
            # )

            # wandb
            plt.savefig("figure.png")
            self.logger.log_image("ValSamples", ["figure.png"], self.current_epoch)

            plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=12, factor=0.5, min_lr=1e-8
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "ValLoss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
