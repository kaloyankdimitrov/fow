import socket
import struct
import time
import collections
import threading
import queue
import random

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining

# ─── Configuration ────────────────────────────────────────────────────────
UDP_IP = "0.0.0.0" # Listen on all interfaces
UDP_PORT = 5005

# VideoMAE Constants
MODEL_ID      = "MCG-NJU/videomae-base"
NUM_FRAMES    = 16
TUBELET_SIZE  = 2
IMG_SIZE      = 224
PATCH_SIZE    = 16
GRID          = IMG_SIZE // PATCH_SIZE
T_TOKENS      = NUM_FRAMES // TUBELET_SIZE
N_TOKENS      = T_TOKENS * GRID * GRID
MASK_RATIO    = 0.75
N_MASKED      = int(N_TOKENS * MASK_RATIO)
N_VISIBLE     = N_TOKENS - N_MASKED
PATCH_DIM     = PATCH_SIZE * PATCH_SIZE * TUBELET_SIZE * 3

IN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IN_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GREY_FILL = 128

# ─── VideoMAE Helpers (From previous script) ──────────────────────────────

def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IN_MEAN, device=t.device).view(3,1,1)
    std  = torch.tensor(IN_STD,  device=t.device).view(3,1,1)
    img  = (t * std + mean).clamp(0, 1).permute(1, 2, 0)
    return (img.cpu().numpy() * 255).astype(np.uint8)

def make_noise_mask(seed: int | None = None) -> torch.Tensor:
    rng = random.Random(seed)
    order = list(range(N_TOKENS))
    rng.shuffle(order)
    noise = torch.zeros(1, N_TOKENS)
    for idx in order[:N_MASKED]:
        noise[0, idx] = 1.0
    return noise

def patchify_video(pixel_values: torch.Tensor) -> torch.Tensor:
    B, C, T, H, W = pixel_values.shape
    p, ts = PATCH_SIZE, TUBELET_SIZE
    gt, gh, gw = T // ts, H // p, W // p
    x = pixel_values.reshape(B, C, gt, ts, gh, p, gw, p)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)
    return x.reshape(B, gt * gh * gw, ts * p * p * C)

def unpatchify_video(patches: torch.Tensor) -> torch.Tensor:
    B, N, D = patches.shape
    p, ts, gt, gh, gw, C = PATCH_SIZE, TUBELET_SIZE, T_TOKENS, GRID, GRID, 3
    x = patches.reshape(B, gt, gh, gw, ts, p, p, C)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return x.reshape(B, C, gt * ts, gh * p, gw * p)

def _apply_spatial_mask(img: np.ndarray, spatial_mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    for idx in range(GRID * GRID):
        if spatial_mask[idx]:
            row, col = divmod(idx, GRID)
            r0, r1 = row * PATCH_SIZE, (row + 1) * PATCH_SIZE
            c0, c1 = col * PATCH_SIZE, (col + 1) * PATCH_SIZE
            out[r0:r1, c0:c1] = GREY_FILL
    return out

# ─── Model Wrapper ────────────────────────────────────────────────────────

class VideoMAEReconstructor:
    def __init__(self, device: str):
        self.device = device
        print(f"[VideoMAE] Loading model on {device}...")
        self.processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
        self.model = VideoMAEForPreTraining.from_pretrained(MODEL_ID).eval().to(device)
        print("[VideoMAE] Ready.")

    @torch.no_grad()
    def reconstruct(self, frames_rgb: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pil_list = [Image.fromarray(f) for f in frames_rgb]
        inputs = self.processor(pil_list, return_tensors="pt")
        pv = inputs["pixel_values"].to(self.device)
        pv_BCTHW = pv.permute(0, 2, 1, 3, 4)

        noise = make_noise_mask().to(self.device)
        bool_masked = (noise > 0.5).to(self.device)

        outputs = self.model(pixel_values=pv, bool_masked_pos=bool_masked)
        logits = outputs.logits

        # Apply per-channel un-normalization fix
        all_patches = patchify_video(pv_BCTHW)
        patches_c = all_patches.view(1, all_patches.shape[1], -1, 3)
        
        patch_mean = patches_c.mean(dim=2, keepdim=True)
        patch_var  = patches_c.var(dim=2, keepdim=True, unbiased=False)
        patch_std  = (patch_var + 1e-6).sqrt()

        logits_c = logits.view(1, logits.shape[1], -1, 3)
        masked_indices = bool_masked.nonzero(as_tuple=True)[0]
        
        pred_unnorm_c = logits_c * patch_std[:, masked_indices, :, :] + patch_mean[:, masked_indices, :, :]
        pred_unnorm = pred_unnorm_c.view(1, -1, PATCH_DIM)

        composite = all_patches[0].clone()
        composite[bool_masked] = pred_unnorm[0]
        clip_recon = unpatchify_video(composite.unsqueeze(0))

        last_orig  = pv_BCTHW[0, :, -1, :, :]
        last_recon = clip_recon[0, :, -1, :, :]

        orig_uint8  = tensor_to_uint8(last_orig)
        recon_uint8 = tensor_to_uint8(last_recon)

        t_idx = T_TOKENS - 1
        start = t_idx * GRID * GRID
        end   = start + GRID * GRID
        spatial_mask = bool_masked[start:end].cpu().numpy()

        masked_uint8 = _apply_spatial_mask(orig_uint8, spatial_mask)
        return orig_uint8, masked_uint8, recon_uint8

class InferenceWorker(threading.Thread):
    def __init__(self, reconstructor: VideoMAEReconstructor):
        super().__init__(daemon=True)
        self.recon = reconstructor
        self.in_q = queue.Queue(maxsize=2)
        self.out_q = queue.Queue(maxsize=1)

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
            try:
                result = self.recon.reconstruct(frames)
                try: self.out_q.get_nowait()
                except queue.Empty: pass
                self.out_q.put(result)
            except Exception as exc:
                print(f"[Model Error] {exc}")

# ─── UI Helpers ───────────────────────────────────────────────────────────

def draw_label(img, text, color=(255, 255, 255)):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    return out

# ─── Main UDP Receiver Loop ───────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reconstructor = VideoMAEReconstructor(device)
    worker = InferenceWorker(reconstructor)
    worker.start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.01) # Non-blocking so cv2.imshow can update

    print(f"[Receiver] Listening on {UDP_IP}:{UDP_PORT} ...")

    frame_buffer = collections.deque(maxlen=NUM_FRAMES)
    assembler = {} # dict to hold chunk data: frame_id -> {chunk_idx: data}
    
    last_result = None
    WIN = "VideoMAE UDP Receiver"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, IMG_SIZE * 3 + 8, IMG_SIZE)

    while True:
        try:
            packet, addr = sock.recvfrom(65535)
            
            # Unpack 8-byte header
            header = packet[:8]
            chunk_data = packet[8:]
            frame_id, total_chunks, chunk_idx = struct.unpack('<IHH', header)
            
            if frame_id not in assembler:
                assembler[frame_id] = {}
            assembler[frame_id][chunk_idx] = chunk_data
            
            # Check if frame is fully assembled
            if len(assembler[frame_id]) == total_chunks:
                # Reassemble in order
                full_data = b''.join([assembler[frame_id][i] for i in range(total_chunks)])
                del assembler[frame_id] # Cleanup
                
                # Cleanup old partial frames to prevent memory leaks
                old_keys = [k for k in assembler.keys() if k < frame_id - 5]
                for k in old_keys: del assembler[k]

                # Decode JPEG
                np_arr = np.frombuffer(full_data, np.uint8)
                bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frame_buffer.append(rgb)
                    
                    if len(frame_buffer) == NUM_FRAMES:
                        worker.push(list(frame_buffer))

        except socket.timeout:
            pass # Expected, just update the GUI
            
        result = worker.latest()
        if result is not None:
            last_result = result

        if last_result is not None:
            original, masked, recon = last_result
            
            orig_bgr = draw_label(cv2.cvtColor(original, cv2.COLOR_RGB2BGR), "UDP Input", (180, 255, 180))
            mask_bgr = draw_label(cv2.cvtColor(masked, cv2.COLOR_RGB2BGR), f"Masked ({MASK_RATIO*100}%)", (100, 100, 255))
            rec_bgr  = draw_label(cv2.cvtColor(recon, cv2.COLOR_RGB2BGR), "Reconstruction", (80, 200, 255))
            
            divider = np.full((IMG_SIZE, 4, 3), 60, dtype=np.uint8)
            panel = np.concatenate([orig_bgr, divider, mask_bgr, divider, rec_bgr], axis=1)
            cv2.imshow(WIN, panel)
        else:
            panel = np.full((IMG_SIZE, IMG_SIZE * 3 + 8, 3), 40, dtype=np.uint8)
            cv2.putText(panel, f"Waiting for UDP stream... Buffered: {len(frame_buffer)}/{NUM_FRAMES}", 
                        (20, IMG_SIZE // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
            cv2.imshow(WIN, panel)

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    sock.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()