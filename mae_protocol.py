"""Wire format for masked VideoMAE clips (RGB uint8, visible patches only)."""
import struct
import zlib

import numpy as np

NUM_FRAMES = 16
IMG_SIZE = 224
PATCH_SIZE = 16
TUBELET_SIZE = 2
GRID = IMG_SIZE // PATCH_SIZE
N_TOKENS = NUM_FRAMES // TUBELET_SIZE * GRID * GRID
PATCH_DIM = TUBELET_SIZE * PATCH_SIZE * PATCH_SIZE * 3
MASK_RATIO = 0.75
N_MASKED = int(N_TOKENS * MASK_RATIO)
HEADER = struct.Struct('<IHH')
MAGIC = b'MAE1'
MASK_BYTES = N_TOKENS // 8
MAX_RAW_SIZE = MASK_BYTES + N_MASKED * 6 * 4 + (N_TOKENS - N_MASKED) * PATCH_DIM


def patchify(frames):
    return frames.reshape(8, 2, GRID, 16, GRID, 16, 3).transpose(
        0, 2, 4, 1, 3, 5, 6).reshape(N_TOKENS, PATCH_DIM)


def unpatchify(patches):
    return patches.reshape(8, GRID, GRID, 2, 16, 16, 3).transpose(
        0, 3, 1, 4, 2, 5, 6).reshape(NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)


def encode_clip(frames, rng=None):
    frames = np.asarray(frames)
    if frames.shape != (NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3) or frames.dtype != np.uint8:
        raise ValueError('Expected 16 RGB uint8 frames of size 224x224')
    rng = np.random.default_rng() if rng is None else rng
    mask = np.zeros(N_TOKENS, dtype=bool)
    mask[rng.choice(N_TOKENS, N_MASKED, replace=False)] = True
    patches = patchify(frames)
    channels = patches[mask].reshape(N_MASKED, -1, 3).astype(np.float32) / 255
    mean = channels.mean(axis=1)
    std = channels.std(axis=1, ddof=1) + 1e-6
    stats = np.stack((mean, std), axis=1).astype('<f4')
    raw = np.packbits(mask).tobytes() + stats.tobytes() + patches[~mask].tobytes()
    return MAGIC + zlib.compress(raw)


def decode_clip(data):
    if not data.startswith(MAGIC):
        raise ValueError('Expected MAE1 masked clip; update sender_mae.py too')
    decoder = zlib.decompressobj()
    raw = decoder.decompress(data[len(MAGIC):], MAX_RAW_SIZE + 1)
    if len(raw) != MAX_RAW_SIZE or not decoder.eof or decoder.unused_data:
        raise ValueError('Invalid masked clip length')
    mask = np.unpackbits(np.frombuffer(raw[:MASK_BYTES], np.uint8)).astype(bool)
    if mask.sum() != N_MASKED:
        raise ValueError('Invalid mask count')
    stats_end = MASK_BYTES + N_MASKED * 6 * 4
    stats = np.frombuffer(raw[MASK_BYTES:stats_end], '<f4').reshape(N_MASKED, 2, 3).copy()
    if not np.isfinite(stats).all() or (stats[:, 1] <= 0).any():
        raise ValueError('Invalid patch statistics')
    patches = np.full((N_TOKENS, PATCH_DIM), 128, dtype=np.uint8)
    patches[~mask] = np.frombuffer(raw[stats_end:], np.uint8).reshape(-1, PATCH_DIM)
    return unpatchify(patches), mask, stats


def packetize(data, clip_id, payload_size=1400):
    count = (len(data) + payload_size - 1) // payload_size
    for index in range(count):
        yield HEADER.pack(clip_id, count, index) + data[index * payload_size:(index + 1) * payload_size]
