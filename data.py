import itertools
import math
import multiprocessing
import random
from collections import Counter

import numpy as np
import pytorch_lightning as pl
import scipy
import torch
from pyannote.audio.core.io import Audio, AudioFile
from pyannote.audio.utils.protocol import check_protocol
from pyannote.core import Segment, SlidingWindow
from scipy.signal import convolve
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data._utils.collate import default_collate


class TrainDataset(IterableDataset):
    def __init__(self, task):
        super().__init__()
        self.task = task

    def __iter__(self):
        return self.task.train__iter__()

    def __len__(self):
        return self.task.train__len__()


class ValDataset(Dataset):
    def __init__(self, task):
        super().__init__()
        self.task = task

    def __getitem__(self, idx):
        return self.task.val__getitem__(idx)

    def __len__(self):
        return self.task.val__len__()


class SegmentationAndSCDData(pl.LightningDataModule):
    def __init__(
        self,
        protocol,
        duration=5.0,
        sample_rate=16_000,
        max_num_speakers=4,
        batch_size=32,
        num_workers=None,
        epoch_length_scaler=4,
        collar=6,  # 293/5=x/0.1
        scd=False,
    ):
        super().__init__()
        self.protocol, self.has_validation = check_protocol(protocol)

        self.sample_rate = sample_rate
        self.scd = scd
        self.epoch_length_scaler = epoch_length_scaler
        self.collar = collar
        self.scd_expansion_window = scipy.signal.triang(self.collar)[:, np.newaxis]
        self.duration = duration
        self.batch_size = batch_size

        self.num_workers = num_workers
        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count() // 2

        self.max_num_speakers = max_num_speakers

    def train_dataloader(self):
        return DataLoader(
            TrainDataset(self),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.has_validation:
            return DataLoader(
                ValDataset(self),
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=False,
                collate_fn=self.collate_fn,
            )
        else:
            return None

    def setup(self, stage=None):
        self.audio = Audio(sample_rate=self.sample_rate, mono=True)

        # Prepare training data
        self._train = []
        for f in self.protocol.train():
            file = dict()
            for key, value in f.items():
                # remove segments shorter than chunks from "annotated" entry
                if key == "annotated":
                    value = [
                        segment for segment in value if segment.duration > self.duration
                    ]
                    file["_annotated_duration"] = sum(
                        segment.duration for segment in value
                    )
                file[key] = value
            self._train.append(file)

        self.determine_max_num_speakers()

        # Prepare validation data
        if not self.has_validation:
            return

        self._validation = []
        for f in self.protocol.development():
            for segment in f["annotated"]:
                if segment.duration < self.duration:
                    continue

                num_chunks = round(segment.duration // self.duration)

                for c in range(num_chunks):
                    start_time = segment.start + c * self.duration
                    chunk = Segment(start_time, start_time + self.duration)
                    self._validation.append((f, chunk))

    def determine_max_num_speakers(self):
        # max_num_speakers already determined to be 4, so only recompute if value
        # not passed to __init__.
        if self.max_num_speakers is None:
            # slide a window (with 1s step) over the whole training set
            # and keep track of the number of speakers in each location
            num_speakers = []
            for file in self._train:
                start = file["annotated"][0].start
                end = file["annotated"][-1].end
                window = SlidingWindow(
                    start=start, end=end, duration=self.duration, step=1.0,
                )
                for chunk in window:
                    num_speakers.append(len(file["annotation"].crop(chunk).labels()))

            # because there might a few outliers, estimate the upper bound for the
            # number of speakers as the 99th percentile

            num_speakers, counts = zip(*list(Counter(num_speakers).items()))
            num_speakers, counts = np.array(num_speakers), np.array(counts)

            sorting_indices = np.argsort(num_speakers)
            num_speakers = num_speakers[sorting_indices]
            counts = counts[sorting_indices]

            self.max_num_speakers = max(
                2,
                num_speakers[np.where(np.cumsum(counts) / np.sum(counts) > 0.99)[0][0]],
            )

    def adapt_y(self, collated_y):
        """Only keep max_num_speakers most talkative speakers per sample.
        Based on https://github.com/pyannote/pyannote-audio/blob/3147e2bfe9a7af388d0c01f3bba3d0578ba60c67/pyannote/audio/tasks/segmentation/segmentation.py#L184.
        """

        batch_size, num_frames, _ = collated_y.shape

        # maximum number of active speakers in a chunk
        max_num_speakers = torch.max(
            torch.sum(torch.sum(collated_y, dim=1) > 0.0, dim=1)
        )

        # sort speakers in descending talkativeness order
        indices = torch.argsort(torch.sum(collated_y, dim=1), dim=1, descending=True)

        # keep max_num_speakers most talkative speakers, for each chunk
        y = torch.zeros(
            (batch_size, num_frames, max_num_speakers), dtype=collated_y.dtype
        )
        for b, index in enumerate(indices):
            for k, i in zip(range(max_num_speakers), index):
                y[b, :, k] = collated_y[b, :, i.item()]

        return y

    def prepare_chunk(self, file: AudioFile, chunk) -> dict:
        """Get audio waveform and corresponding labels. Returns a dictionary with key X mapped
        to the waveform of shape (num_samples, num_channels) and key y mapped to a
        SlidingWindowFeature of shape (num_frames, num_labels)
        """

        sample = dict()

        # read (and resample if needed) audio chunk
        sample["X"], _ = self.audio.crop(file, chunk, duration=self.duration)

        resolution = self.duration / self.num_frames

        # Discretize annotation: Convert from continuous to discrete num_frames accepted by model.
        sample["y"] = file["annotation"].discretize(
            support=chunk, resolution=resolution, duration=self.duration
        )

        return sample

    def scd_postprocess_y(self, Y):
        """Generate labels for speaker change detection. Y is a discretized annotation.
        Returns a numpy array of shape (num_samples, 1).
        """
        # replace NaNs by 0s
        Y = np.nan_to_num(Y)

        y = np.zeros((Y.shape[0], 1))
        speaker = Y[0] if np.sum(Y[0]) > 0 else np.array([0] * y.shape[1])
        for idx, row in enumerate(Y):
            if np.sum(row) > 0:
                new_speaker = row
                if (new_speaker != speaker).any():
                    y[idx] = 1
                speaker = new_speaker

        # mark change points neighborhood as positive
        y = np.minimum(1, convolve(y, self.scd_expansion_window, mode="same"))

        y = 1 * (y > 1e-10)

        return y

    def scd_prepare_chunk(self, file, chunk):
        sample = dict()

        # read (and resample if needed) audio chunk
        sample["X"], _ = self.audio.crop(file, chunk, duration=self.duration)

        resolution = self.duration / self.num_frames
        y = file["annotation"].discretize(
            support=chunk, resolution=resolution, duration=self.duration
        )

        sample["y"] = self.scd_postprocess_y(y)
        return sample

    def train__iter__(self):
        train = self._train

        while True:
            # select one file at random (with probability proportional to its annotated duration)
            file, *_ = random.choices(
                train, weights=[f["_annotated_duration"] for f in train], k=1,
            )

            # select one annotated region at random (with probability proportional to its duration)
            segment, *_ = random.choices(
                file["annotated"], weights=[s.duration for s in file["annotated"]], k=1,
            )

            # select one chunk at random (with uniform distribution)
            start_time = random.uniform(segment.start, segment.end - self.duration)
            chunk = Segment(start_time, start_time + self.duration)

            if self.scd:
                yield self.scd_prepare_chunk(file, chunk)
            else:
                yield self.prepare_chunk(file, chunk)

    def collate_y(self, batch):
        # gather common set of labels
        # b["y"] is a SlidingWindowFeature instance
        labels = sorted(set(itertools.chain(*(b["y"].labels for b in batch))))

        batch_size, num_frames, num_labels = (
            len(batch),
            len(batch[0]["y"]),
            len(labels),
        )
        Y = np.zeros((batch_size, num_frames, num_labels), dtype=np.int64)

        for i, b in enumerate(batch):
            for local_idx, label in enumerate(b["y"].labels):
                global_idx = labels.index(label)
                Y[i, :, global_idx] = b["y"].data[:, local_idx]

        return torch.from_numpy(Y)

    def collate_fn(self, batch):
        collated_X = default_collate([b["X"] for b in batch])

        if self.scd:
            collated_y = default_collate([b["y"] for b in batch])
        else:
            collated_y = self.collate_y(batch)

        y = collated_y if self.scd else self.adapt_y(collated_y)
        return {"X": collated_X, "y": y}

    def train__len__(self):
        # Number of training samples in one epoch
        duration = sum(file["_annotated_duration"] for file in self._train)
        return (
            max(self.batch_size, math.ceil(duration / self.duration))
            * self.epoch_length_scaler
        )

    def val__getitem__(self, idx):
        f, chunk = self._validation[idx]
        if self.scd:
            return self.scd_prepare_chunk(f, chunk)
        else:
            return self.prepare_chunk(f, chunk)

    def val__len__(self):
        return len(self._validation)
