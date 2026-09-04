"""The U-Net that reconstructs dense fragment-centre coverage from sparse input.

One class covers all three variants that were run on the cluster:

    use_sequence=True,  dual_stream=False    V1, and V2 with sequence   8,517,953 params
    use_sequence=False, dual_stream=False    V2 without sequence        8,515,393
    dual_stream=True                         V3, two input streams      8,545,665

V1 and V2-with-sequence are the same network by construction, so a V1 checkpoint
loads into V2 and vice versa. Nothing in a .ckpt records which variant wrote it;
the parameter counts above are the only discriminator, and they only separate
with-sequence from without.

SUBMODULE NAMES ARE FROZEN. `enc1`, `bottleneck`, `dec3`, `output`,
`coverage_stream` and the rest reproduce the attribute names in
`old_nucae/model_v{1,2,3}.py`, so state_dict keys match and the existing cluster
checkpoints load with strict=True. Renaming any of them silently breaks that.

RECEPTIVE FIELD, MEASURED. Backpropagating from one output position and counting
non-zero gradient support on the input gives 2,073 bp with the GroupNorms
replaced by Identity -- about 11 nucleosome repeats, not the 337 bp quoted in the
thesis draft (that figure counts one convolution per ResBlock and weights the
stages by the wrong stride). With the GroupNorms active the support is the whole
input, because GroupNorm reduces over channels AND positions, so every output
depends on every input through the normalisation statistics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Input length must be divisible by 8 (the encoder's total stride) and greater
# than 192: reflect padding requires the pad to be smaller than the dimension it
# pads, and the widest dilation pads by 24 at the bottleneck, where the signal is
# L/8 long. Irrelevant at WINDOW, but it bites when testing on short windows.
WINDOW = 25_000               # bp per window
N_NT = 5                      # one-hot A, C, G, T, N
KERNEL = 7
DILATIONS = (1, 2, 4, 8)      # spans (k-1)*d = 6, 12, 24, 48 bp: local, TF footprint, sub-nucleosomal
NORM_GROUPS = 8               # GroupNorm, not BatchNorm: batch is small at L=25,000
BASE_CHANNELS = 64            # then 128, 256 -- halve length, double depth, constant capacity
STREAM_CHANNELS = 32          # per input stream in the dual-stream variant
HELICAL_KERNEL = 11           # one full DNA helical turn (~10 bp), for the sequence stream


def _same_padding(dilation: int = 1, kernel: int = KERNEL) -> int:
    """Padding that preserves length for an odd, symmetric kernel."""
    return dilation * (kernel - 1) // 2


class ResBlock(nn.Module):
    """Four dilated convolutions plus a shortcut.

    Reflect padding rather than zeros: a coverage window has no natural boundary,
    and zero-padding would present the edge as an abrupt drop to no coverage.
    GELU rather than ReLU because the signal is centred near zero after the
    running-median subtraction, where ReLU would suppress half the activations.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 dilations: tuple[int, ...] = DILATIONS) -> None:
        super().__init__()
        self.block = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(in_channels if i == 0 else out_channels, out_channels,
                          kernel_size=KERNEL, padding=_same_padding(d), dilation=d,
                          padding_mode="reflect"),
                nn.GroupNorm(NORM_GROUPS, out_channels),
                nn.GELU(),
            )
            for i, d in enumerate(dilations)
        )
        self.residual = (nn.Conv1d(in_channels, out_channels, kernel_size=1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.block:
            out = layer(out)
        return out + self.residual(x)


class EncoderStage(nn.Module):
    """ResBlock at full resolution, then a strided convolution that halves length."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.resblock = ResBlock(in_channels, out_channels)
        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=KERNEL,
                                    stride=2, padding=_same_padding())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.resblock(x)
        return skip, self.downsample(skip)


class DecoderStage(nn.Module):
    """Upsample, concatenate the encoder skip, then a ResBlock.

    Linear interpolation followed by a convolution, not a transposed convolution:
    uneven kernel overlap in the latter puts periodic checkerboard artefacts into
    the output, which is unacceptable when the quantity of interest is itself a
    periodicity.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=KERNEL,
                              padding=_same_padding())
        self.resblock = ResBlock(2 * out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
        return self.resblock(torch.cat([self.conv(x), skip], dim=1))


class CoverageStream(nn.Module):
    """Dedicated first convolution for the coverage channel (V3)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, STREAM_CHANNELS, kernel_size=KERNEL,
                               padding=_same_padding())
        self.norm = nn.GroupNorm(NORM_GROUPS, STREAM_CHANNELS)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(self.norm(self.conv1(x)))


class SequenceStream(nn.Module):
    """Dedicated first convolution for the one-hot sequence (V3).

    A 2D kernel of height N_NT spans all four bases plus N at once, so a filter
    sees a base composition rather than four independent channels. Its width of
    11 bp is the smallest odd kernel covering one DNA helical turn, which is the
    period at which flexible (WW) and stiff (SS) dinucleotides recur at
    rotationally equivalent positions on the nucleosome surface.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, STREAM_CHANNELS, kernel_size=(N_NT, HELICAL_KERNEL),
                               padding=(0, _same_padding(kernel=HELICAL_KERNEL)))
        self.norm = nn.GroupNorm(NORM_GROUPS, STREAM_CHANNELS)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x.unsqueeze(1))          # (B, 5, L) -> (B, 32, 1, L)
        return self.gelu(self.norm(x.squeeze(2)))


class FragmentomicsUNet(nn.Module):
    """Three encoder stages, a bottleneck, three decoder stages, a 1x1 head.

    Input  (B, 6, L) -- channel 0 coverage, channels 1-5 one-hot sequence --
    or (B, 1, L) when use_sequence is False. Output (B, 1, L), no activation.

    With dual_stream the coverage and sequence are convolved separately and
    concatenated before the encoder, rather than entering one shared convolution
    that has to serve both.
    """

    def __init__(self, use_sequence: bool = True, dual_stream: bool = False) -> None:
        super().__init__()
        if dual_stream and not use_sequence:
            raise ValueError("dual_stream reads the sequence channels; "
                             "use_sequence=False leaves it nothing to read")
        self.dual_stream = dual_stream

        if dual_stream:
            self.coverage_stream = CoverageStream()
            self.sequence_stream = SequenceStream()
            enc1_in = 2 * STREAM_CHANNELS
        else:
            enc1_in = 1 + N_NT if use_sequence else 1

        self.enc1 = EncoderStage(enc1_in, BASE_CHANNELS)
        self.enc2 = EncoderStage(BASE_CHANNELS, 2 * BASE_CHANNELS)
        self.enc3 = EncoderStage(2 * BASE_CHANNELS, 4 * BASE_CHANNELS)
        self.bottleneck = ResBlock(4 * BASE_CHANNELS, 4 * BASE_CHANNELS)
        self.dec3 = DecoderStage(4 * BASE_CHANNELS, 4 * BASE_CHANNELS)
        self.dec2 = DecoderStage(4 * BASE_CHANNELS, 2 * BASE_CHANNELS)
        self.dec1 = DecoderStage(2 * BASE_CHANNELS, BASE_CHANNELS)
        self.output = nn.Conv1d(BASE_CHANNELS, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dual_stream:
            x = torch.cat([self.coverage_stream(x[:, :1, :]),
                           self.sequence_stream(x[:, 1:, :])], dim=1)

        skip1, x = self.enc1(x)                 # skip1 (B,  64, L),   x (B,  64, L/2)
        skip2, x = self.enc2(x)                 # skip2 (B, 128, L/2), x (B, 128, L/4)
        skip3, x = self.enc3(x)                 # skip3 (B, 256, L/4), x (B, 256, L/8)

        x = self.bottleneck(x)

        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.output(x)
