import math

import networkx as nx
import numpy as np
import torch
from einops import rearrange
from pyannote.audio.core.io import Audio
from pyannote.audio.utils.permutation import mae_cost_func, permutate
from pyannote.audio.utils.signal import Binarize, binarize
from pyannote.core import SlidingWindow, SlidingWindowFeature


class Inference:
    def __init__(
        self,
        model,
        duration=5,
        step=None,
        batch_size=32,
        sample_rate=16_000,
        scd=False,
    ):

        self.model = model
        self.model.eval()

        self.stitch_threshold = 0.39

        self.duration = duration
        self.scd = scd

        # Step between consecutive chunks
        if step is None:
            step = 0.1 * self.duration

        self.step = step

        self.batch_size = batch_size
        self.sample_rate = sample_rate

    def slide(self, waveform, sample_rate=16_000):
        """
        Slide model on a waveform and generate predictions. Returns a SlidingWindowFeature.
        """

        window_size: int = round(self.duration * sample_rate)
        step_size: int = round(self.step * sample_rate)

        # waveform is of shape [num_channels=1, num_samples]

        # Split the waveform into 5 second (80,000 samples) chunks. The next chunk is generated
        # `step_size=8,000` samples from the start of the previous chunk. So, for 60 seconds of
        # audio (960,000 samples), chunks has shape [num_chunks=111, num_channels=1, num_frames=80,000]
        chunks = rearrange(
            waveform.unfold(1, window_size, step_size),
            "channel chunk frame -> chunk channel frame",
        )
        num_chunks, _, _ = chunks.shape

        outputs = []

        # Predict speaker activations using model for a batch of chunks.
        for c in np.arange(0, num_chunks, self.batch_size):
            batch = chunks[c : c + self.batch_size]
            with torch.no_grad():
                outputs.append(self.model(batch).cpu().numpy())

        outputs = np.vstack(outputs)

        frames = SlidingWindow(start=0.0, duration=self.duration, step=self.step)
        return SlidingWindowFeature(outputs, frames)

    def get_stitchable_components(
        self, segmentations, stitch_threshold, onset=0.5
    ) -> nx.Graph:
        """Build stitching graph.
        Taken from https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/pipelines/segmentation.py#L139.
        
        Parameters
        ----------
        segmentations : (num_chunks, num_frames, local_num_speakers)-shaped SlidingWindowFeature
            Raw output of segmentation model.
        onset : float, optional
            Onset speaker activation threshold. Defaults to 0.5

        Returns
        -------
        stitching_graph : nx.Graph
            Nodes are (chunk_idx, speaker_idx) tuples.
            An edge between two nodes indicate that those are likely to be the same speaker
            (the lower the value of "cost" attribute, the more likely).
        """

        chunks = segmentations.sliding_window
        num_chunks, num_frames, _ = segmentations.data.shape
        max_lookahead = math.floor(chunks.duration / chunks.step - 1)
        lookahead = 2 * (max_lookahead,)

        stitching_graph = nx.Graph()

        for C, (chunk, segmentation) in enumerate(segmentations):
            for c in range(
                max(0, C - lookahead[0]), min(num_chunks, C + lookahead[1] + 1)
            ):

                if c == C:
                    continue

                # extract common temporal support
                shift = round((C - c) * num_frames * chunks.step / chunks.duration)

                if shift < 0:
                    shift = -shift
                    this_segmentations = segmentation[shift:]
                    that_segmentations = segmentations[c, : num_frames - shift]
                else:
                    this_segmentations = segmentation[: num_frames - shift]
                    that_segmentations = segmentations[c, shift:]

                # find the optimal one-to-one mapping
                _, (permutation,), (cost,) = permutate(
                    this_segmentations[np.newaxis],
                    that_segmentations,
                    cost_func=mae_cost_func,
                    return_cost=True,
                )

                for this, that in enumerate(permutation):

                    this_is_active = np.any(this_segmentations[:, this] > onset)
                    that_is_active = np.any(that_segmentations[:, that] > onset)

                    if this_is_active:
                        stitching_graph.add_node((C, this))

                    if that_is_active:
                        stitching_graph.add_node((c, that))

                    if this_is_active and that_is_active:
                        stitching_graph.add_edge(
                            (C, this), (c, that), cost=cost[this, that]
                        )

        # A component is 'stitchable' if it contains at most one node per chunk
        f = stitching_graph.copy()
        while f:
            f.remove_edges_from(
                [
                    (n1, n2)
                    for n1, n2, cost in f.edges(data="cost")
                    if cost > stitch_threshold
                ]
            )
            for component in list(nx.connected_components(f)):
                if len(set(c for c, _ in component)) == len(component):
                    yield component
                    f.remove_nodes_from(component)
            stitch_threshold *= 0.5

    def aggregate_combine(
        self, segmentations, count,
    ):
        """Aggregate and combine speakers using preprocessed segmentation and precomputed speaker count.
        Outputs a binary SlidingWindowFeature.
        """
        activations = self.aggregate(
            segmentations,
            frames=count.sliding_window,
            hamming=False,
            missing=0.0,
            skip_average=True,
        )

        _, num_speakers = activations.data.shape
        count.data = np.minimum(count.data, num_speakers)

        extent = activations.extent & count.extent
        activations = activations.crop(extent, return_data=False)
        count = count.crop(extent, return_data=False)

        sorted_speakers = np.argsort(-activations, axis=-1)
        binary = np.zeros_like(activations.data)

        for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers)):
            for i in range(c.item()):
                binary[t, speakers[i]] = 1.0

        return SlidingWindowFeature(binary, activations.sliding_window), activations

    def speaker_count(
        self, segmentations, onset=0.5, offset=None, frames=None,
    ) -> SlidingWindowFeature:
        """Returns a SlidingWindowFeature of shape (num_frames, 1) with the estimated
        number of speakers per frame.
        """

        binarized = binarize(
            segmentations, onset=onset, offset=offset, initial_state=False
        )
        count = self.aggregate(
            np.sum(binarized, axis=-1, keepdims=True),
            frames=frames,
            hamming=False,
            missing=0.0,
            skip_average=False,
        )
        count.data = np.rint(count.data).astype(np.uint8)

        return count

    def __call__(self, file, return_scd_points=False):
        """
        Run inference on a whole file (type is AudioFile).
        Output is a tuple of (Annotation of segmented speakers, Aggregated model activations).
        Set `return_scd_points` to return points of speaker changing if in SCD mode.
        """

        audio = Audio(sample_rate=self.sample_rate, mono=True)

        waveform, sample_rate = audio(file)

        # Slide across the audio file to create a SWF of shape
        # [num_chunks, num_model_features=293, num_speakers=4].
        # There are num_model_features speaker predictions per chunk (5s).
        segmentations = self.slide(waveform, sample_rate)

        if not self.scd:
            # Build stitching graph: Take the segmentations and build a graph where
            # the nodes are (chunk_idx, speaker_idx) tuples and the edges are the likelihood
            # that nodes represent the same speaker. Can be thought of as placing the graphs
            # of two chunks on top of each other and then mapping speaker activations
            # based on how close every line in one chunk is to every line in the other chunk.
            # Get connected components that contain only one node per chunk because we
            # want to stitch chunks together.
            components = list(
                self.get_stitchable_components(
                    segmentations, stitch_threshold=self.stitch_threshold
                )
            )

            num_stitches = len(components)
            num_chunks, num_frames, _ = segmentations.data.shape

            stitched_segmentations = np.NAN * np.zeros(
                (num_chunks, num_frames, num_stitches)
            )

            # Realign the same speakers to be on the same axis.
            for k, component in enumerate(components):
                for chunk_idx, speaker_idx in component:
                    stitched_segmentations[chunk_idx, :, k] = segmentations.data[
                        chunk_idx, :, speaker_idx
                    ]

            stitched_segmentations = SlidingWindowFeature(
                stitched_segmentations, segmentations.sliding_window
            )

            segmentations = stitched_segmentations

        count = self.speaker_count(
            segmentations, frames=None,  # used to be self._frames
        )

        discrete_segmentations, activations = self.aggregate_combine(
            segmentations, count
        )

        # Remove inactive speakers
        discrete_segmentations.data = discrete_segmentations.data[
            :, np.nonzero(np.sum(discrete_segmentations.data, axis=0))[0]
        ]

        # Binarize detection scores and convert to Annotation
        new_binarize = Binarize(
            onset=0.5, offset=0.5, min_duration_on=0, min_duration_off=0,
        )

        final_output = new_binarize(discrete_segmentations)

        # Compute timestamps of speaker change if directly performing SCD
        # and return_scd_points is true.
        if self.scd and return_scd_points:
            change_points = [segment.middle for segment, _ in final_output.itertracks()]
            return change_points, activations

        return final_output, activations

    @staticmethod
    def aggregate(
        scores: SlidingWindowFeature,
        frames: SlidingWindow = None,
        epsilon: float = 1e-12,
        hamming: bool = False,
        missing: float = np.NaN,
        skip_average: bool = False,
    ) -> SlidingWindowFeature:
        """Aggregation
        From https://github.com/pyannote/pyannote-audio/blob/3147e2bfe9a7af388d0c01f3bba3d0578ba60c67/pyannote/audio/core/inference.py#L411

        Parameters
        ----------
        scores : SlidingWindowFeature
            Raw (unaggregated) scores. Shape is (num_chunks, num_frames_per_chunk, num_classes).
        frames : SlidingWindow, optional
            Frames resolution. Defaults to estimate it automatically based on `scores` shape
            and chunk size. Providing the exact frame resolution (when known) leads to better
            temporal precision.
        warm_up : (float, float) tuple, optional
            Left/right warm up duration (in seconds).
        missing : float, optional
            Value used to replace missing (ie all NaNs) values.
        skip_average : bool, optional
            Skip final averaging step.

        Returns
        -------
        aggregated_scores : SlidingWindowFeature
            Aggregated scores. Shape is (num_frames, num_classes)
        """

        num_chunks, num_frames_per_chunk, num_classes = scores.data.shape

        chunks = scores.sliding_window
        if frames is None:
            duration = step = chunks.duration / num_frames_per_chunk
            frames = SlidingWindow(start=chunks.start, duration=duration, step=step)
        else:
            frames = SlidingWindow(
                start=chunks.start, duration=frames.duration, step=frames.step,
            )

        masks = 1 - np.isnan(scores)
        scores.data = np.nan_to_num(scores.data, copy=True, nan=0.0)

        # Hamming window used for overlap-add aggregation
        hamming_window = (
            np.hamming(num_frames_per_chunk).reshape(-1, 1)
            if hamming
            else np.ones((num_frames_per_chunk, 1))
        )

        # aggregated_output[i] will be used to store the sum of all predictions
        # for frame #i
        num_frames = (
            frames.closest_frame(
                scores.sliding_window.start
                + scores.sliding_window.duration
                + (num_chunks - 1) * scores.sliding_window.step
            )
            + 1
        )
        aggregated_output: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # overlapping_chunk_count[i] will be used to store the number of chunks
        # that contributed to frame #i
        overlapping_chunk_count: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # aggregated_mask[i] will be used to indicate whether
        # at least one non-NAN frame contributed to frame #i
        aggregated_mask: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # loop on the scores of sliding chunks
        for (chunk, score), (_, mask) in zip(scores, masks):
            # chunk ~ Segment
            # score ~ (num_frames_per_chunk, num_classes)-shaped np.ndarray
            # mask ~ (num_frames_per_chunk, num_classes)-shaped np.ndarray

            start_frame = frames.closest_frame(chunk.start)
            aggregated_output[start_frame : start_frame + num_frames_per_chunk] += (
                score * mask * hamming_window
            )

            overlapping_chunk_count[
                start_frame : start_frame + num_frames_per_chunk
            ] += (mask * hamming_window)

            aggregated_mask[
                start_frame : start_frame + num_frames_per_chunk
            ] = np.maximum(
                aggregated_mask[start_frame : start_frame + num_frames_per_chunk], mask,
            )

        if skip_average:
            average = aggregated_output
        else:
            average = aggregated_output / np.maximum(overlapping_chunk_count, epsilon)

        average[aggregated_mask == 0.0] = missing

        return SlidingWindowFeature(average, frames)
