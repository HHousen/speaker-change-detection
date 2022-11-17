import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid_filterbanks import Encoder, ParamSincFB


# https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/models/blocks/sincnet.py
class SincNet(nn.Module):
    def __init__(self, sample_rate: int = 16000, stride: int = 1):
        super(SincNet, self).__init__()

        self.stride = stride

        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)

        self.conv1d = nn.ModuleList()
        self.pool1d = nn.ModuleList()
        self.norm1d = nn.ModuleList()

        self.conv1d.append(
            Encoder(
                ParamSincFB(
                    80,
                    251,
                    stride=self.stride,
                    sample_rate=sample_rate,
                    min_low_hz=50,
                    min_band_hz=50,
                )
            )
        )
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(80, affine=True))

        self.conv1d.append(nn.Conv1d(80, 60, 5, stride=1))
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(60, affine=True))

        self.conv1d.append(nn.Conv1d(60, 60, 5, stride=1))
        self.pool1d.append(nn.MaxPool1d(3, stride=3, padding=0, dilation=1))
        self.norm1d.append(nn.InstanceNorm1d(60, affine=True))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward
        Parameters
        ----------
        waveforms : (batch, channel, sample)
        """

        outputs = self.wav_norm1d(waveforms)

        for c, (conv1d, pool1d, norm1d) in enumerate(
            zip(self.conv1d, self.pool1d, self.norm1d)
        ):

            outputs = conv1d(outputs)

            # https://github.com/mravanelli/SincNet/issues/4
            if c == 0:
                outputs = torch.abs(outputs)

            outputs = F.leaky_relu(norm1d(pool1d(outputs)))

        return outputs
