import cv2
import torch
import socket
import pickle
import struct
import time
from transformers import AutoImageProcessor, ViTMAEForPreTraining

# --- Configuration ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
CHUNK_SIZE = 1000  # Safely under maximum UDP payload size

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load Pre-trained Open-Weight MAE
processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(device)
model.eval()

# Socket Setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# OpenCV Video Capture (Source 1 as specified)
cap = cv2.VideoCapture(1)

print("Sender online. Streaming compressed latents...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # MAE expects RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb_frame, return_tensors="pt").to(device)

    with torch.no_grad():
        # The vit backbone automatically drops 75% of tokens by default
        vit_outputs = model.vit(inputs.pixel_values)
        
        # Extract latents (1, 50, 768) and restoration maps (1, 196)
        latent = vit_outputs.last_hidden_state.cpu()
        ids_restore = vit_outputs.ids_restore.cpu()

    # Flatten the original image into patches to compute local stats
    # Assuming an input tensor 'pixel_values' of shape (1, 3, 224, 224)
    B, C, H, W = inputs.pixel_values.shape
    p = 16 # Patch size

    # Reshape into blocks of patches: (1, 196, 768)
    patches = inputs.pixel_values.reshape(B, C, H//p, p, W//p, p)
    patches = patches.permute(0, 2, 4, 1, 3, 5).flatten(1, 2).flatten(2)

    # Compute per-patch mean and standard deviation
    patch_means = patches.mean(dim=-1, keepdim=True).cpu()
    patch_stds = patches.std(dim=-1, keepdim=True).cpu()

    # Serialize and transmit (latent, ids_restore, patch_means, patch_stds)
    payload = pickle.dumps((latent, ids_restore, patch_means, patch_stds))
    
    # Fragment and transmit over UDP
    total_chunks = (len(payload) + CHUNK_SIZE - 1) // CHUNK_SIZE
    frame_id = int(time.time() * 1000) % 100000
    
    for chunk_idx in range(total_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, len(payload))
        chunk_data = payload[start:end]
        
        # Pack header: Frame ID (4B), Chunk Index (2B), Total Chunks (2B)
        header = struct.pack("!IHH", frame_id, chunk_idx, total_chunks)
        sock.sendto(header + chunk_data, (UDP_IP, UDP_PORT))

    # Control frame rate roughly to match network stability (~30 FPS)
    time.sleep(0.03)

cap.release()