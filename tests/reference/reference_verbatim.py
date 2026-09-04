"""
pipeline.py
-----------
Reads GC-corrected FCC counts, applies full smoothing pipeline,
writes normalised signal to a new output file.

Usage:
    python pipeline.py <input_tsv_gz> <output_tsv_gz> [--signal_col 3]
"""

import gzip
import sys
import argparse
import numpy as np
import scipy.ndimage
from scipy.signal import medfilt
from whittaker_eilers import WhittakerSmoother


def smooth_and_normalise(signal: np.ndarray) -> np.ndarray:
    """Apply full pipeline: Whittaker + Gaussian + running median."""
    # Step 1: Whittaker-Eilers smoothing
    smoother = WhittakerSmoother(lmbda=1000, order=2, data_length=len(signal))
    signal   = np.array(smoother.smooth(signal))

    # Step 2: Gaussian smoothing
    signal = scipy.ndimage.gaussian_filter1d(signal, sigma=30)

    # Step 3: Running median normalisation
    running_med = medfilt(signal, kernel_size=375)
    signal      = signal - running_med

    return signal.astype(np.float32)


def process_chromosome(input_path: str, output_path: str, signal_col: int = 3):
    """
    Read GC-corrected counts, apply pipeline, write normalised signal.
    Output: chr  position  normalised_signal  (3 columns)
    """
    print(f'  Reading {input_path}...')
    chroms    = []
    positions = []
    signal    = []

    with gzip.open(input_path, 'rt') as f:
        for line in f:
            parts = line.split('\t')
            chroms.append(parts[0])
            positions.append(int(parts[1]))
            signal.append(float(parts[signal_col]))

    signal = np.array(signal, dtype=np.float64)
    print(f'  {len(signal):,} positions loaded  '
          f'signal range [{signal.min():.4f}, {signal.max():.4f}]')

    print('  Applying smoothing pipeline...')
    normalised = smooth_and_normalise(signal)

    print(f'  Writing to {output_path}...')
    with gzip.open(output_path, 'wt') as f:
        for chrom, pos, val in zip(chroms, positions, normalised):
            f.write(f'{chrom}\t{pos}\t{val:.6f}\n')

    print(f'  Done. Normalised range [{normalised.min():.4f}, {normalised.max():.4f}]')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input',       help='Input TSV.gz (GC-corrected counts)')
    parser.add_argument('output',      help='Output TSV.gz (normalised signal)')
    parser.add_argument('--signal_col', type=int, default=3,
                        help='Column index of GC-corrected signal (default: 3)')
    args = parser.parse_args()

    process_chromosome(args.input, args.output, args.signal_col)


if __name__ == '__main__':
    main()
