import socket
import time
import threading
import queue
import zlib

import cv2
import numpy as np
import torch
from transformers import VideoMAEForPreTraining
from mae_protocol import (HEADER as PACKET_HEADER, IMG_SIZE, MASK_RATIO,
                          N_MASKED, PATCH_DIM, decode_clip, patchify, unpatchify)

# ─── Configuration ────────────────────────────────────────────────────────
UDP_IP = "0.0.0.0" # Listen on all interfaces
UDP_PORT = 5005
FRAME_TIMEOUT = 3.0
MAX_PENDING_FRAMES = 32

# VideoMAE Constants
MODEL_ID      = "MCG-NJU/videomae-base"
IN_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IN_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ─── Model Wrapper ────────────────────────────────────────────────────────

class VideoMAEReconstructor:
    def __init__(self, device: str):
        self.device = device
        print(f"[VideoMAE] Loading model on {device}...")
        self.model = VideoMAEForPreTraining.from_pretrained(MODEL_ID).eval().to(device)
        print("[VideoMAE] Ready.")

    @torch.no_grad()
    def reconstruct(self, clip) -> tuple[np.ndarray, np.ndarray]:
        frames, mask, stats = clip
        # The wire carries only visible pixels. Normalize them for the encoder;
        # masked tubelets are excluded by bool_masked_pos inside VideoMAE.
        rgb = frames.astype(np.float32) / 255
        normalized = (rgb - IN_MEAN) / IN_STD
        pv = torch.from_numpy(normalized).permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
        bool_masked = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        logits = self.model(pixel_values=pv, bool_masked_pos=bool_masked).logits
        predictions = logits[0].cpu().numpy().reshape(N_MASKED, -1, 3)
        if self.model.config.norm_pix_loss:
            predictions = predictions * stats[:, 1, None, :] + stats[:, 0, None, :]
        patches = patchify(rgb).copy()
        patches[mask] = predictions.reshape(N_MASKED, PATCH_DIM)
        reconstructed = (unpatchify(patches).clip(0, 1) * 255).astype(np.uint8)
        return frames[-1], reconstructed[-1]

class InferenceWorker(threading.Thread):
    def __init__(self, reconstructor: VideoMAEReconstructor):
        super().__init__(daemon=True)
        self.recon = reconstructor
        self.in_q = queue.Queue(maxsize=2)
        self.out_q = queue.Queue(maxsize=1)

    def push(self, frames):
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

class FrameAssembler:
    """Reassemble sender_mae's masked clip chunks, bounding incomplete frame storage."""

    def __init__(self):
        self.pending = {}

    def expire(self):
        cutoff = time.monotonic() - FRAME_TIMEOUT
        for key, (created, _, _) in list(self.pending.items()):
            if created < cutoff:
                del self.pending[key]

    def add(self, packet, addr):
        self.expire()
        if len(packet) <= PACKET_HEADER.size:
            return None
        frame_id, total_chunks, chunk_idx = PACKET_HEADER.unpack_from(packet)
        if total_chunks == 0 or chunk_idx >= total_chunks:
            return None

        key = (addr, frame_id)
        if key not in self.pending:
            if len(self.pending) >= MAX_PENDING_FRAMES:
                del self.pending[next(iter(self.pending))]
            self.pending[key] = (time.monotonic(), total_chunks, {})
        _, expected, chunks = self.pending[key]
        if total_chunks != expected:
            return None
        chunks[chunk_idx] = packet[PACKET_HEADER.size:]
        if len(chunks) == expected:
            del self.pending[key]
            return b''.join(chunks[i] for i in range(expected))
        return None

class UDPWorker(threading.Thread):
    """Drain UDP independently of GUI event processing and model inference."""

    def __init__(self, sock):
        super().__init__(daemon=True)
        self.sock = sock
        self.clips = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.packets = 0
        self.completed = 0
        self.errors = 0
        self.assembler = FrameAssembler()

    def run(self):
        while not self.stop_event.is_set():
            try:
                packet, addr = self.sock.recvfrom(65535)
            except socket.timeout:
                self.assembler.expire()
                continue
            except OSError:
                if self.stop_event.is_set():
                    break
                raise
            self.packets += 1
            data = self.assembler.add(packet, addr)
            if data is None:
                continue
            try:
                clip = decode_clip(data)
            except (ValueError, zlib.error) as exc:
                self.errors += 1
                print(f"[Packet Error] {exc}")
                continue
            self.completed += 1
            try:
                self.clips.get_nowait()
            except queue.Empty:
                pass
            self.clips.put_nowait(clip)

    def latest(self):
        try:
            return self.clips.get_nowait()
        except queue.Empty:
            return None

    def status(self):
        return f"Packets: {self.packets}  Clips: {self.completed}  Errors: {self.errors}"


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Some operating systems cap receive buffers below this requested size.
    # A dedicated reader is necessary even with a large buffer.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.1)
    network = UDPWorker(sock)
    network.start()
    print(f"[Receiver] Listening on {UDP_IP}:{UDP_PORT} ...")

    WIN = "VideoMAE UDP Receiver"
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        reconstructor = VideoMAEReconstructor(device)
        worker = InferenceWorker(reconstructor)
        worker.start()
        last_result = None
        masked_input = None
        last_status = time.monotonic()
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, IMG_SIZE * 2 + 4, IMG_SIZE)

        while network.is_alive():
            clip = network.latest()
            if clip is not None:
                masked_input = clip[0][-1]
                worker.push(clip)
            result = worker.latest()
            if result is not None:
                last_result = result

            if last_result is not None or masked_input is not None:
                if last_result is not None:
                    masked, recon = last_result
                    rec_label = "Reconstruction"
                else:
                    masked = masked_input
                    recon = np.full_like(masked, 40)
                    rec_label = "Reconstructing..."
                mask_bgr = draw_label(cv2.cvtColor(masked, cv2.COLOR_RGB2BGR),
                                      f"Sender masked ({MASK_RATIO*100}%)", (100, 100, 255))
                rec_bgr = draw_label(cv2.cvtColor(recon, cv2.COLOR_RGB2BGR),
                                     rec_label, (80, 200, 255))
                divider = np.full((IMG_SIZE, 4, 3), 60, dtype=np.uint8)
                panel = np.concatenate([mask_bgr, divider, rec_bgr], axis=1)
            else:
                panel = np.full((IMG_SIZE, IMG_SIZE * 2 + 4, 3), 40, dtype=np.uint8)
                cv2.putText(panel, "Waiting for masked UDP clip...", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(panel, network.status(), (4, IMG_SIZE - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            cv2.imshow(WIN, panel)
            if time.monotonic() - last_status >= 5:
                print(f"[Receiver] {network.status()}")
                last_status = time.monotonic()
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        network.stop_event.set()
        network.join(timeout=1)
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
