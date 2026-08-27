import cv2
import numpy as np
import socket

# --- Configuration ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
PATCH_SIZE = 16
GRID_SIZE = 14
MASK_RATIO = 0.75  # Correctly configured: Hides 75% of the frame data

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Fixed: Instantiated cleanly once to maximize frame rate
cap = cv2.VideoCapture(1)

print(f"Streaming high-performance sparse video to {UDP_IP}:{UDP_PORT}...")

frame_id = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        # Center-crop and scale to target network size
        h, w, _ = frame.shape
        crop = min(h, w)
        cropped = frame[(h-crop)//2:(h+crop)//2, (w-crop)//2:(w+crop)//2]
        resized = cv2.resize(cropped, (224, 224))
        
        # Seed the generator deterministically using our moving frame ID
        np.random.seed(frame_id)
        total_patches = GRID_SIZE * GRID_SIZE
        num_keep = int(total_patches * (1 - MASK_RATIO))
        shuffled_indices = np.random.permutation(total_patches)
        keep_indices = shuffled_indices[:num_keep]
        
        # Build sparse canvas
        masked_frame = np.zeros_like(resized)
        for idx in keep_indices:
            i, j = divmod(idx, GRID_SIZE)
            y1, y2 = i * PATCH_SIZE, (i + 1) * PATCH_SIZE
            x1, x2 = j * PATCH_SIZE, (j + 1) * PATCH_SIZE
            masked_frame[y1:y2, x1:x2] = resized[y1:y2, x1:x2]
        
        # Encode canvas as a highly optimized JPEG
        success, encoded_frame = cv2.imencode('.jpg', masked_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success: 
            continue
            
        # Pack the 4-byte frame ID ahead of the image bytes
        # This completely avoids sending heavy mask data over the network!
        payload = frame_id.to_bytes(4, byteorder='big') + encoded_frame.tobytes()
        
        try:
            sock.sendto(payload, (UDP_IP, UDP_PORT))
        except OSError:
            pass
            
        cv2.imshow("Sender Output (Sparse Stream)", masked_frame)
        frame_id = (frame_id + 1) % 1000000  # Safe boundary reset
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
finally:
    cap.release()
    cv2.destroyAllWindows()