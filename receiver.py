import socket
import pickle
import cv2
import numpy as np
import torch
# Swap AutoImageProcessor for ViTImageProcessor
from transformers import ViTImageProcessor, ViTMAEForPreTraining

# --- Configuration ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
PATCH_SIZE = 16
GRID_SIZE = 14

# Initialize Model explicitly
print("Loading Local ViT-MAE for real-time inference...")
model_id = "facebook/vit-mae-large"
processor = ViTImageProcessor.from_pretrained(model_id) # Using explicit processor
model = ViTMAEForPreTraining.from_pretrained(model_id, device_map="mps")
model.eval()

# Set up receiver socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

print(f"Receiver listening on port {UDP_PORT}...")

try:
    while True:
        # Grab packet data
        packet_data, _ = sock.recvfrom(65536)
        if not packet_data:
            continue
            
        try:
            payload = pickle.loads(packet_data)
        except Exception:
            continue # Skip corrupted or incomplete UDP frame streams
            
        keep_indices = payload["indices"]
        visible_patches = payload["patches"]
        
        # Reconstruct the partial "received" frame canvas
        received_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Re-build custom explicit mask tensor for MAE tracking
        # 1 = masked patch, 0 = visible patch
        custom_mask = np.ones((GRID_SIZE * GRID_SIZE), dtype=np.float32)
        
        for idx, patch in zip(keep_indices, visible_patches):
            i, j = divmod(idx, GRID_SIZE)
            received_frame[i*PATCH_SIZE:(i+1)*PATCH_SIZE, j*PATCH_SIZE:(j+1)*PATCH_SIZE] = patch
            custom_mask[idx] = 0.0
            
        # Run standard inference on the structured frame
        rgb_frame = cv2.cvtColor(received_frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb_frame, return_tensors="pt").to(model.device)
        
        # Format custom mask to fit matching model batch structure
        custom_mask_tensor = torch.from_numpy(custom_mask).unsqueeze(0).to(model.device)
        
        with torch.no_grad():
            # Injecting explicit layout structures bypassing internal random noise masking loops
            outputs = model(**inputs, noise=custom_mask_tensor) 
            logits = outputs.logits
            
        # Map tokens back into pixel array
        reconstructed_pixels = model.unpatchify(logits).detach().cpu().squeeze()
        reconstructed_img = reconstructed_pixels.permute(1, 2, 0).numpy()
        
        # De-normalize 
        reconstructed_img = reconstructed_img * np.array(processor.image_std) + np.array(processor.image_mean)
        reconstructed_img = np.clip(reconstructed_img * 255, 0, 255).astype(np.uint8)
        reconstructed_bgr = cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR)
        
        # View Results
        display_output = np.hstack((received_frame, reconstructed_bgr))
        cv2.imshow("Receiver (Left: Received Patches | Right: AI Restored Output)", cv2.resize(display_output, (896, 448)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    sock.close()