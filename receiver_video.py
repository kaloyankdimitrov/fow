"""Reconstruct sender_video.py's sparse JPEG stream with VideoMAE."""
import collections
import queue
import socket
import threading

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, VideoMAEForPreTraining

UDP_IP = "0.0.0.0"
UDP_PORT = 5005
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_FRAMES = 16
TUBELET_SIZE = 2
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
MODEL_ID = "MCG-NJU/videomae-base"


def load_repatched_videomae(model_name=MODEL_ID, device="cpu"):
    # Let Transformers handle checkpoint conversion for its own model version.
    # In particular, VideoMAE's key projection does not have a bias parameter.
    model = VideoMAEForPreTraining.from_pretrained(model_name).eval().to(device)
    return model, AutoImageProcessor.from_pretrained(model_name)


def patchify_videomae(pixel_values, config):
    batch, frames, channels, height, width = pixel_values.shape
    p, u = config.patch_size, config.tubelet_size
    x = pixel_values.reshape(batch, frames // u, u, channels,
                             height // p, p, width // p, p)
    return x.permute(0, 1, 4, 6, 2, 5, 7, 3).reshape(batch, -1, u * p * p * channels)


def unpatchify_videomae(patches, config):
    batch = patches.shape[0]
    p, u, channels = config.patch_size, config.tubelet_size, config.num_channels
    grid = config.image_size // p
    frames = patches.shape[1] // (grid * grid) * u
    x = patches.reshape(batch, frames // u, grid, grid, u, p, p, channels)
    return x.permute(0, 1, 4, 7, 2, 5, 3, 6).reshape(
        batch, frames, channels, config.image_size, config.image_size)


def detect_mask(frame):
    patches = frame.reshape(GRID_SIZE, PATCH_SIZE, GRID_SIZE, PATCH_SIZE, 3)
    return patches.mean(axis=(1, 3, 4)) < 15.0


def expand_mask(mask):
    return np.repeat(np.repeat(mask, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1)


class VideoReconstructor:
    def __init__(self, model, processor, device):
        self.model = model
        self.device = device
        self.mean = torch.tensor(processor.image_mean, device=device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(processor.image_std, device=device).view(1, 1, 3, 1, 1)
        self.history = collections.deque(maxlen=NUM_FRAMES)

    @torch.inference_mode()
    def reconstruct(self, incoming):
        mask = detect_mask(incoming)
        pixel_mask = expand_mask(mask)
        if not self.history:
            # Bootstrap prior context without treating black holes as known pixels.
            reference = cv2.inpaint(incoming, pixel_mask.astype(np.uint8) * 255,
                                    3, cv2.INPAINT_TELEA)
            self.history.extend(reference.copy() for _ in range(NUM_FRAMES - 1))
        if not mask.any():
            self.history.append(incoming.copy())
            return incoming.copy()

        frames = list(self.history)[-(NUM_FRAMES - 1):] + [incoming]
        raw = torch.from_numpy(np.stack(frames)).to(self.device, dtype=torch.float32)
        raw = raw.permute(0, 3, 1, 2).unsqueeze(0) / 255.0
        token_mask = torch.zeros((1, NUM_FRAMES // TUBELET_SIZE, GRID_SIZE, GRID_SIZE),
                                 device=self.device, dtype=torch.bool)
        token_mask[0, -1] = torch.from_numpy(mask).to(self.device)
        token_mask = token_mask.flatten(1)
        logits = self.model((raw - self.mean) / self.std,
                            bool_masked_pos=token_mask).logits

        # True missing-pixel statistics are unavailable in this protocol. Estimate
        # them from repaired history, retaining current pixels wherever received.
        reference = raw.clone()
        missing = torch.from_numpy(pixel_mask).to(self.device)
        reference[0, -1, :, missing] = raw[0, -2, :, missing]
        patches = patchify_videomae(raw, self.model.config)
        predicted = logits[0].reshape(-1, TUBELET_SIZE * PATCH_SIZE * PATCH_SIZE, 3)
        if self.model.config.norm_pix_loss:
            ref_patches = patchify_videomae(reference, self.model.config)[token_mask]
            ref_channels = ref_patches.reshape(-1, TUBELET_SIZE * PATCH_SIZE * PATCH_SIZE, 3)
            mean = ref_channels.mean(dim=1, keepdim=True)
            std = ref_channels.var(dim=1, keepdim=True, unbiased=True).sqrt() + 1e-6
            predicted = predicted * std + mean
        patches[token_mask] = predicted.reshape(-1, patches.shape[-1])
        restored = unpatchify_videomae(patches, self.model.config)[0, -1]
        restored = (restored.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        output = incoming.copy()
        output[pixel_mask] = restored[pixel_mask]
        self.history.append(output.copy())
        return output


def put_latest(q, value):
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    q.put_nowait(value)


class InferenceWorker(threading.Thread):
    def __init__(self, reconstructor):
        super().__init__(daemon=True)
        self.reconstructor = reconstructor
        self.inputs = queue.Queue(maxsize=1)
        self.outputs = queue.Queue(maxsize=1)
        self.error = None

    def run(self):
        while True:
            frame = self.inputs.get()
            if frame is None:
                return
            try:
                output = self.reconstructor.reconstruct(frame)
                put_latest(self.outputs, (frame, output))
                self.error = None
            except Exception as exc:
                self.error = str(exc)
                print(f"[Model Error] {exc}")


def main():
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model, processor = load_repatched_videomae(device=device)
    worker = InferenceWorker(VideoReconstructor(model, processor, device))
    worker.start()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    window = "VideoMAE: received / reconstructed"
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        sock.bind((UDP_IP, UDP_PORT))
        sock.setblocking(False)
        print(f"Receiver listening on UDP {UDP_IP}:{UDP_PORT}")
        panel = np.full((IMAGE_SIZE, IMAGE_SIZE * 2, 3), 40, dtype=np.uint8)
        cv2.putText(panel, "Waiting for sender_video.py...", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        while True:
            latest_packet = None
            # Bound draining so a continuous stream cannot starve GUI events.
            for _ in range(64):
                try:
                    latest_packet, _ = sock.recvfrom(65535)
                except BlockingIOError:
                    break
            if latest_packet:
                decoded = cv2.imdecode(np.frombuffer(latest_packet, np.uint8), cv2.IMREAD_COLOR)
                if decoded is not None:
                    incoming = cv2.cvtColor(cv2.resize(decoded, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2RGB)
                    put_latest(worker.inputs, incoming)
            try:
                incoming, restored = worker.outputs.get_nowait()
                panel = cv2.cvtColor(np.hstack((incoming, restored)), cv2.COLOR_RGB2BGR)
            except queue.Empty:
                pass
            display = panel.copy()
            if worker.error:
                cv2.putText(display, "Model error: see terminal", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 1)
            cv2.imshow(window, display)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        put_latest(worker.inputs, None)
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
