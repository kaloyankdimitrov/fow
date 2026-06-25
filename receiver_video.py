import socket
import cv2
import numpy as np
import torch
from transformers import ViTImageProcessor, ViTMAEForPreTraining

# --- Configuration ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
WIDTH, HEIGHT = 224, 224
PATCH_SIZE = 16
GRID_SIZE = 14

# Automatically leverage Apple Silicon (MPS) or CUDA if available
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

print(f"Using device: {device}")
print("Loading stable ViT-MAE Model (No missing weights)...")
processor = ViTImageProcessor.from_pretrained("facebook/vit-mae-base")
model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(device)
model.eval()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Core Temporal Memory Anchor
previous_reconstructed_frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

print("Listening for incoming sparse video frames...")

try:
    while True:
        packet, _ = sock.recvfrom(65536)
        if not packet: continue
        
        # 1. Decode incoming JPEG packet from sender
        frame_arr = np.frombuffer(packet, dtype=np.uint8)
        incoming_frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
        if incoming_frame is None: continue
            
        # Create a copy that we will enrich with temporal data
        blended_frame = incoming_frame.copy()
        
        # 2. TEMPORAL MEMORY FILL: Swap out sender-masked black squares with history
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                y1, y2 = i * PATCH_SIZE, (i + 1) * PATCH_SIZE
                x1, x2 = j * PATCH_SIZE, (j + 1) * PATCH_SIZE
                
                patch = incoming_frame[y1:y2, x1:x2]
                # If the patch was blacked out by the sender, recover it from the last frame
                if np.all(patch == 0):
                    blended_frame[y1:y2, x1:x2] = previous_reconstructed_frame[y1:y2, x1:x2]

        # 3. Transformer Processing (Refining edges, lighting shifts, and new objects)
        rgb_frame = cv2.cvtColor(blended_frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=rgb_frame, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # unpatchify converts token logits back into a [1, 3, 224, 224] image tensor
            reconstructed_tensor = model.unpatchify(outputs.logits)
            
        # 4. Format tensor back to standard displayable BGR image
        rec_img = reconstructed_tensor.detach().cpu().squeeze().permute(1, 2, 0).numpy()
        
        # Reverse ImageNet normalization applied by the processor
        imagenet_mean = np.array(processor.image_mean)
        imagenet_std = np.array(processor.image_std)
        rec_img = np.clip((rec_img * imagenet_std + imagenet_mean) * 255, 0, 255).astype(np.uint8)
        final_frame = cv2.cvtColor(rec_img, cv2.COLOR_RGB2BGR)
        
        # 5. Commit the output frame to memory for the next incoming frame
        previous_reconstructed_frame = final_frame.copy()
        
        # Render Comparison Displays
        cv2.imshow("1. Incoming Sparse Stream (Network)", cv2.resize(incoming_frame, (448, 448)))
        cv2.imshow("2. Real-Time Temporal Reconstruction", cv2.resize(final_frame, (448, 448)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cv2.destroyAllWindows()
    sock.close()