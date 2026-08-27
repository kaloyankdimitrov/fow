import cv2
import socket
import struct
import time
import math

# ─── Configuration ────────────────────────────────────────────────────────
UDP_IP = "127.0.0.1"  # Change to receiver's IP if on a different machine
UDP_PORT = 5005
MAX_PAYLOAD = 1400    # Safe size below standard 1500 MTU
IMG_SIZE = 224
JPEG_QUALITY = 80

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"[Sender] Streaming to {UDP_IP}:{UDP_PORT} ... Press Ctrl+C to stop.")
    
    frame_id = 0
    
    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
                
            # Resize and compress
            resized = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            success, encoded = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not success:
                continue
                
            data = encoded.tobytes()
            total_chunks = math.ceil(len(data) / MAX_PAYLOAD)
            
            # Packet Header: Frame ID (unsigned int), Total Chunks (unsigned short), Chunk Index (unsigned short)
            # Format '<IHH' = 4 bytes + 2 bytes + 2 bytes = 8 bytes total
            for chunk_idx in range(total_chunks):
                start = chunk_idx * MAX_PAYLOAD
                end = start + MAX_PAYLOAD
                chunk_data = data[start:end]
                
                header = struct.pack('<IHH', frame_id, total_chunks, chunk_idx)
                packet = header + chunk_data
                sock.sendto(packet, (UDP_IP, UDP_PORT))
                
            frame_id = (frame_id + 1) % 4294967295 # Wrap around max unsigned int
            
            # Cap sender framerate slightly to avoid saturating local network buffers
            time.sleep(0.03) 
            
    except KeyboardInterrupt:
        print("\n[Sender] Stopped.")
    finally:
        cap.release()
        sock.close()

if __name__ == "__main__":
    main()