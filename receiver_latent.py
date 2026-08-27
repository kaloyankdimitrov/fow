import cv2
import torch
import socket
import pickle
import struct
import numpy as np
from collections import defaultdict
from transformers import ViTMAEForPreTraining

# --- Configuration ---
UDP_IP = "0.0.0.0"
UDP_PORT = 5005

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(device)
model.eval()

# Socket Setup
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print("Receiver online. Listening for incoming latents...")

frame_buffer = defaultdict(dict)
# ImageNet normalization constants used by the model
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

while True:
    packet, addr = sock.recvfrom(65535)
    if len(packet) < 8:
        continue
        
    # Unpack frame fragmentation header
    header = packet[:8]
    chunk_data = packet[8:]
    frame_id, chunk_idx, total_chunks = struct.unpack("!IHH", header)
    
    frame_buffer[frame_id][chunk_idx] = chunk_data
    
    # Periodic buffer clearing to prevent memory leak on dropped frames
    if len(frame_buffer) > 20:
        frame_buffer.clear()
        
    # Once all chunks for a specific frame arrive, reassemble
    if len(frame_buffer[frame_id]) == total_chunks:
        full_payload = b"".join([frame_buffer[frame_id][i] for i in range(total_chunks)])
        del frame_buffer[frame_id]
        
        try:
            latent, ids_restore, patch_means, patch_stds = pickle.loads(full_payload)
            latent = latent.to(device)
            ids_restore = ids_restore.to(device)
            patch_means = patch_means.to(device)
            patch_stds = patch_stds.to(device)

            with torch.no_grad():
                # Pass the truncated latents + mapping to the decoder
                decoder_outputs = model.decoder(latent, ids_restore)
                # decoder_outputs.logits shape is (1, 196, 768)
                pred_logits = decoder_outputs.logits.cpu()

                # Un-normalize each patch using its exact corresponding transmitted stat
                unnormalized_patches = pred_logits * patch_stds + patch_means

# Proceed to reshape and stitch via unpatchify...
                
            # Unpatchify: Stitches 14x14 grid of 16x16 patches back into 224x224 image
            p, h, w = 16, 14, 14
            x = unnormalized_patches.reshape(1, h, w, p, p, 3)
            x = x.permute(0, 1, 3, 2, 4, 5)  # Align pixel sequences spatially
            rec_img = x.reshape(h * p, w * p, 3).numpy()
            
            # De-normalize back to traditional RGB range
            rec_img = rec_img * std + mean
            rec_img = np.clip(rec_img * 255, 0, 255).astype(np.uint8)
            
            # Convert RGB to BGR for OpenCV rendering
            bgr_frame = cv2.cvtColor(rec_img, cv2.COLOR_RGB2BGR)
            bgr_frame = cv2.resize(bgr_frame, (480, 480), interpolation=cv2.INTER_CUBIC)
            
            cv2.imshow("AI Reconstructed Stream (75% Omitted)", bgr_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"Frame drops or corruption encountered: {e}")

cv2.destroyAllWindows()