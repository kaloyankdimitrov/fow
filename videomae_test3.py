"""
VideoMAE live webcam demo — continuous masked video reconstruction.

Architecture
────────────
Model : MCG-NJU/videomae-base  (ViT-B, 16 frames, 224×224)
Tokens: 16 frames / 2 (tubelet_size) = 8 temporal positions
        × (224/16)² = 196 spatial positions
        = 1568 space-time patch tokens per clip
Mask  : 75 % → 1176 tokens hidden, 392 visible
"""

import argparse
import collections
import threading
import time
import random
import queue

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining

# ─── constants ────────────────────────────────────────────────────────────────
MODEL_ID      = "MCG-NJU/videomae-base"
NUM_FRAMES    = 16          # clip length the model expects
TUBELET_SIZE  = 2           # temporal patch depth
IMG_SIZE      = 224         # spatial size
PATCH_SIZE    = 16          # spatial patch edge
GRID          = IMG_SIZE // PATCH_SIZE          # 14
T_TOKENS      = NUM_FRAMES  // TUBELET_SIZE     # 8
N_TOKENS      = T_TOKENS * GRID * GRID          # 1568
MASK_RATIO    = 0.5
N_MASKED      = int(N_TOKENS * MASK_RATIO)      # 1176
N_VISIBLE     = N_TOKENS - N_MASKED             # 392
PATCH_DIM     = PATCH_SIZE * PATCH_SIZE * TUBELET_SIZE * 3  # 1536

# ImageNet stats used by the processor for input normalisation
IN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IN_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

GREY_FILL = 128   # intensity of masked-patch overlay

# ─── helpers ──────────────────────────────────────────────────────────────────

def bgr_to_rgb_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) ImageNet-normalised float → uint8 H×W×3 RGB."""
    mean = torch.tensor(IN_MEAN, device=t.device).view(3,1,1)
    std  = torch.tensor(IN_STD,  device=t.device).view(3,1,1)
    img  = (t * std + mean).clamp(0, 1).permute(1, 2, 0)
    return (img.cpu().numpy() * 255).astype(np.uint8)

def make_noise_mask(seed: int | None = None) -> torch.Tensor:
    """
    Return a (1, N_TOKENS) float noise tensor where masked positions
    have value 1.0 and visible positions 0.0.
    """
    rng = random.Random(seed)
    order = list(range(N_TOKENS))
    rng.shuffle(order)
    noise = torch.zeros(1, N_TOKENS)
    for idx in order[:N_MASKED]:
        noise[0, idx] = 1.0
    return noise   # (1, 1568)

# ─── patchify / unpatchify for VideoMAE tube patches ─────────────────────────

def patchify_video(pixel_values: torch.Tensor) -> torch.Tensor:
    """(B, C, T, H, W) → (B, N, D)"""
    B, C, T, H, W = pixel_values.shape
    p = PATCH_SIZE
    ts = TUBELET_SIZE
    gt = T  // ts          # 8  temporal positions
    gh = H  // p           # 14 spatial rows
    gw = W  // p           # 14 spatial cols

    x = pixel_values.reshape(B, C, gt, ts, gh, p, gw, p)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
    x = x.reshape(B, gt * gh * gw, ts * p * p * C)
    return x

def unpatchify_video(patches: torch.Tensor) -> torch.Tensor:
    """(B, N, D) → (B, C, T, H, W) inverse of patchify_video."""
    B, N, D = patches.shape
    p  = PATCH_SIZE
    ts = TUBELET_SIZE
    gt = T_TOKENS
    gh = gw = GRID
    C  = 3

    x = patches.reshape(B, gt, gh, gw, ts, p, p, C)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)   # (B, C, gt, ts, gh, p, gw, p)
    x = x.reshape(B, C, gt * ts, gh * p, gw * p)
    return x

# ─── core model wrapper ───────────────────────────────────────────────────────

class VideoMAEReconstructor:
    def __init__(self, device: str):
        self.device = device
        print(f"[VideoMAE] Loading {MODEL_ID} on {device} …")
        self.processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
        self.model = VideoMAEForPreTraining.from_pretrained(MODEL_ID)
        self.model.eval().to(device)
        print(f"[VideoMAE] Ready. "
              f"Tokens={N_TOKENS}, masked={N_MASKED} ({MASK_RATIO*100:.0f}%)")

    @torch.no_grad()
    def reconstruct(
        self,
        frames_rgb: list[np.ndarray],   # list of 16 H×W×3 uint8 RGB arrays
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # ── 1. Pre-process: resize + ImageNet-normalise ────────────────
        pil_list = [Image.fromarray(f) for f in frames_rgb]
        inputs = self.processor(pil_list, return_tensors="pt")
        pv = inputs["pixel_values"].to(self.device)

        # Rearrange to (B, C, T, H, W) for our patchify
        pv_BCTHW = pv.permute(0, 2, 1, 3, 4)   # (1, 3, 16, 224, 224)

        # ── 2. Build noise mask ────────────────────────────────────────
        noise = make_noise_mask(seed).to(self.device)   # (1, 1568)

        # ── 3. Forward pass ───────────────────────────────────────────
        bool_masked = (noise > 0.5).to(self.device)   # (1, 1568) bool

        outputs = self.model(
            pixel_values=pv,               
            bool_masked_pos=bool_masked,   
        )
        logits = outputs.logits            # (1, N_MASKED, 1536)

        # ── 4. Decode with per-patch normalisation ─────────────────────
        original_frame, masked_frame, recon_frame = self._decode(
            pv_BCTHW, logits, bool_masked[0]
        )
        return original_frame, masked_frame, recon_frame

    def _decode(
        self,
        pv_BCTHW: torch.Tensor,          # (1, 3, 16, 224, 224) ImageNet-normed
        logits: torch.Tensor,             # (1, N_masked, D)
        bool_masked: torch.Tensor,        # (N_tokens,) bool, True=masked
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        all_patches = patchify_video(pv_BCTHW)  # (1, 1568, 1536)

        # ── THE FIX: Per-Channel Normalisation ──
        # Reshape to isolate the 3 colour channels so they aren't averaged into grey.
        # 1536 = 512 spatial/temporal pixels × 3 channels
        patches_c = all_patches.view(1, all_patches.shape[1], -1, 3)  # (1, N, 512, 3)

        # Compute stats per-channel. unbiased=False strictly matches HF's internals.
        patch_mean = patches_c.mean(dim=2, keepdim=True)                       # (1, N, 1, 3)
        patch_var  = patches_c.var(dim=2, keepdim=True, unbiased=False)        # (1, N, 1, 3)
        patch_std  = (patch_var + 1e-6).sqrt()                                 # (1, N, 1, 3)

        # Reshape logits to match the separated channels
        logits_c = logits.view(1, logits.shape[1], -1, 3)                      # (1, N_masked, 512, 3)

        masked_indices = bool_masked.nonzero(as_tuple=True)[0]                 # (N_masked,)
        
        # Un-normalise per-channel, preserving the true base colour
        pred_unnorm_c = logits_c * patch_std[:, masked_indices, :, :] \
                                 + patch_mean[:, masked_indices, :, :]         # (1, N_masked, 512, 3)

        # Flatten back to standard patch dimension
        pred_unnorm = pred_unnorm_c.view(1, -1, PATCH_DIM)                     # (1, N_masked, 1536)

        # Build composite patch sequence
        composite = all_patches[0].clone()                  # (N, D) mutable copy
        composite[bool_masked] = pred_unnorm[0]             # fill masked slots
        composite = composite.unsqueeze(0)                  # (1, N, D)

        # Reconstruct full clip tensor
        clip_recon = unpatchify_video(composite)            # (1, 3, 16, 224, 224)

        # Extract the LAST frame
        last_orig  = pv_BCTHW[0, :, -1, :, :]   # (3, 224, 224)
        last_recon = clip_recon[0, :, -1, :, :]  # (3, 224, 224)

        orig_uint8  = tensor_to_uint8(last_orig)
        recon_uint8 = tensor_to_uint8(last_recon)

        t_idx   = T_TOKENS - 1                           # 7
        start   = t_idx * GRID * GRID                    # 1372
        end     = start + GRID * GRID                    # 1568
        spatial_mask = bool_masked[start:end].cpu().numpy()  # (196,) bool

        masked_uint8 = _apply_spatial_mask(orig_uint8, spatial_mask)

        return orig_uint8, masked_uint8, recon_uint8

def _apply_spatial_mask(img: np.ndarray, spatial_mask: np.ndarray) -> np.ndarray:
    """Grey out spatially-masked patches on the display frame."""
    out = img.copy()
    for idx in range(GRID * GRID):
        if spatial_mask[idx]:
            row, col = divmod(idx, GRID)
            r0, r1 = row * PATCH_SIZE, (row + 1) * PATCH_SIZE
            c0, c1 = col * PATCH_SIZE, (col + 1) * PATCH_SIZE
            out[r0:r1, c0:c1] = GREY_FILL
    return out

# ─── background inference thread ─────────────────────────────────────────────

class InferenceWorker(threading.Thread):
    def __init__(self, reconstructor: VideoMAEReconstructor):
        super().__init__(daemon=True)
        self.recon  = reconstructor
        self.in_q   = queue.Queue(maxsize=2)
        self.out_q  = queue.Queue(maxsize=1)
        self._frame_count = 0

    def push(self, frames: list[np.ndarray]):
        try:
            self.in_q.put_nowait(frames)
        except queue.Full:
            pass

    def latest(self) -> tuple | None:
        try:
            return self.out_q.get_nowait()
        except queue.Empty:
            return None

    def run(self):
        while True:
            frames = self.in_q.get()
            t0 = time.perf_counter()
            try:
                result = self.recon.reconstruct(frames)
                dt = time.perf_counter() - t0
                self._frame_count += 1
                fps_model = 1.0 / dt if dt > 0 else 0
                print(f"\r[model] frame {self._frame_count:5d}  "
                      f"latency {dt*1000:.0f} ms  ({fps_model:.1f} fps)",
                      end="", flush=True)
                try:
                    self.out_q.get_nowait()
                except queue.Empty:
                    pass
                self.out_q.put(result)
            except Exception as exc:
                print(f"\n[model] ERROR: {exc}")

# ─── overlay helpers ──────────────────────────────────────────────────────────

def draw_label(img: np.ndarray, text: str, colour=(255, 255, 255)) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1, cv2.LINE_AA)
    return out

def make_display(
    original: np.ndarray,
    masked: np.ndarray,
    recon: np.ndarray,
    cam_fps: float,
    buffered: int,
) -> np.ndarray:
    orig_bgr  = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    mask_bgr  = cv2.cvtColor(masked,   cv2.COLOR_RGB2BGR)
    rec_bgr   = cv2.cvtColor(recon,    cv2.COLOR_RGB2BGR)

    orig_bgr = draw_label(orig_bgr,
        f"Original  cam {cam_fps:.1f} fps  buf {buffered}/16",
        (180, 255, 180))
    mask_bgr = draw_label(mask_bgr,
        f"Masked ({int(MASK_RATIO*100)}% hidden  "
        f"{N_VISIBLE} visible / {N_MASKED} hidden)",
        (100, 100, 255))
    rec_bgr  = draw_label(rec_bgr,
        "VideoMAE reconstruction",
        (80, 200, 255))

    divider = np.full((IMG_SIZE, 4, 3), 60, dtype=np.uint8)
    panel = np.concatenate([orig_bgr, divider, mask_bgr, divider, rec_bgr], axis=1)
    return panel

def make_waiting_display(buffered: int, cam_fps: float) -> np.ndarray:
    panel = np.full((IMG_SIZE, IMG_SIZE * 3 + 8, 3), 40, dtype=np.uint8)
    msg = (f"Buffering frames … {buffered}/{NUM_FRAMES}  "
           f"(cam {cam_fps:.1f} fps)")
    cv2.putText(panel, msg,
                (20, IMG_SIZE // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(panel, "Press Q or Esc to quit",
                (20, IMG_SIZE // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)
    return panel

# ─── main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VideoMAE live masked video reconstruction from webcam."
    )
    parser.add_argument("--camera",  type=int,   default=0,
                        help="Webcam device index (default 0)")
    parser.add_argument("--device",  type=str,   default=None,
                        help="torch device: cuda / cpu (auto-detected if omitted)")
    parser.add_argument("--skip",    type=int,   default=1,
                        help="Process every N-th webcam frame (default 1 = all)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print(f"[cam] Opened camera index {args.camera}")

    reconstructor = VideoMAEReconstructor(device)
    worker = InferenceWorker(reconstructor)
    worker.start()

    frame_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=NUM_FRAMES)

    last_result  = None
    cam_fps      = 0.0
    t_fps        = time.perf_counter()
    frame_count  = 0
    skip_counter = 0

    WIN = "VideoMAE  |  Original / Masked / Reconstructed  [Q to quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, IMG_SIZE * 3 + 8, IMG_SIZE)

    print("\n[main] Starting live loop.  Press Q or Esc to quit.\n")

    while True:
        ret, bgr = cap.read()
        if not ret:
            print("[cam] Frame read failed — retrying …")
            time.sleep(0.05)
            continue

        frame_count += 1
        now = time.perf_counter()
        if now - t_fps >= 1.0:
            cam_fps  = frame_count / (now - t_fps)
            frame_count = 0
            t_fps    = now

        resized_bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        frame_buffer.append(rgb)

        if len(frame_buffer) == NUM_FRAMES:
            skip_counter += 1
            if skip_counter >= args.skip:
                skip_counter = 0
                worker.push(list(frame_buffer))

        result = worker.latest()
        if result is not None:
            last_result = result

        buffered = len(frame_buffer)
        if last_result is not None:
            original, masked, recon = last_result
            panel = make_display(original, masked, recon, cam_fps, buffered)
        else:
            panel = make_waiting_display(buffered, cam_fps)

        cv2.imshow(WIN, panel)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[main] Done.")

if __name__ == "__main__":
    main()