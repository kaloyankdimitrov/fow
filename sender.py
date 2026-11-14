from video_source import open_source, parse_source_args

args = parse_source_args()

import cv2
import numpy as np
import socket
import pickle

# --- Configuration ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
PATCH_SIZE = 16
GRID_SIZE = 14  # 14 * 16 = 224
MASK_RATIO = 0.75  

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

cap = open_source(args, fallback_camera=True)

print(f"Sending optimized video patches to {UDP_IP}:{UDP_PORT}...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Center crop & resize to 224x224
        h, w, _ = frame.shape
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]
        resized = cv2.resize(cropped, (224, 224))
        
        # Chop image into a 14x14 grid of 16x16 pixel patches
        patches = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                patch = resized[i*PATCH_SIZE:(i+1)*PATCH_SIZE, j*PATCH_SIZE:(j+1)*PATCH_SIZE]
                patches.append(patch)
        
        patches = np.array(patches) # Shape: (196, 16, 16, 3)
        num_patches = len(patches)
        num_keep = int(num_patches * (1 - MASK_RATIO))
        
        # Randomly choose which patches to keep
        shuffled_indices = np.random.permutation(num_patches)
        keep_indices = shuffled_indices[:num_keep]
        
        # Filter down to ONLY the unmasked patch data
        visible_patches = patches[keep_indices]
        
        # Package only visible data + their original layout positions
        payload = {
            "indices": keep_indices,
            "patches": visible_patches
        }
        
        # Serialize and send
        data = pickle.dumps(payload)
        if len(data) > 65507:
            print("Packet size too large for UDP standard limits! Lower your resolution or increase masking.")
            continue
            
        sock.sendto(data, (UDP_IP, UDP_PORT))

        # Show local preview of what is being skipped vs sent
        mask_visual = np.zeros_like(resized)
        for idx, patch in zip(keep_indices, visible_patches):
            i, j = divmod(idx, GRID_SIZE)
            mask_visual[i*PATCH_SIZE:(i+1)*PATCH_SIZE, j*PATCH_SIZE:(j+1)*PATCH_SIZE] = patch
            
        cv2.imshow("Sender Side (Transmitted Patches Only)", mask_visual)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
