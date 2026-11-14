from video_source import open_source, parse_source_args

args = parse_source_args()

import cv2
import numpy as np
import socket

# --- Configuration ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
PATCH_SIZE = 16
GRID_SIZE = 14
MASK_RATIO = 0.25  # Hide 75% of the frame data

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cap = open_source(args, fallback_camera=True)

print(f"Streaming compressed sparse video to {UDP_IP}:{UDP_PORT}...")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        h, w, _ = frame.shape
        crop = min(h, w)
        cropped = frame[(h-crop)//2:(h+crop)//2, (w-crop)//2:(w+crop)//2]
        resized = cv2.resize(cropped, (224, 224))
        
        # Generate spatial mask indices
        total_patches = GRID_SIZE * GRID_SIZE
        num_keep = int(total_patches * (1 - MASK_RATIO))
        shuffled_indices = np.random.permutation(total_patches)
        keep_indices = shuffled_indices[:num_keep]
        
        # Build masked canvas
        masked_frame = np.zeros_like(resized)
        for idx in keep_indices:
            i, j = divmod(idx, GRID_SIZE)
            y1, y2 = i * PATCH_SIZE, (i + 1) * PATCH_SIZE
            x1, x2 = j * PATCH_SIZE, (j + 1) * PATCH_SIZE
            masked_frame[y1:y2, x1:x2] = resized[y1:y2, x1:x2]
        
        # Encode whole canvas as a singular efficient JPEG
        success, encoded_frame = cv2.imencode('.jpg', masked_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success: continue
            
        payload = encoded_frame.tobytes()
        try:
            sock.sendto(payload, (UDP_IP, UDP_PORT))
        except OSError:
            pass
            
        cv2.imshow("Sender Output", masked_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release()
    cv2.destroyAllWindows()
