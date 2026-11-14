import cv2
from video_source import open_source, parse_source_args
import socket
import time
from mae_protocol import NUM_FRAMES, encode_clip, packetize

# ─── Configuration ────────────────────────────────────────────────────────
UDP_IP = "127.0.0.1"  # Change to receiver's IP if on a different machine
UDP_PORT = 5005
MAX_PAYLOAD = 1400    # Safe size below standard 1500 MTU
IMG_SIZE = 224

def main():
    args = parse_source_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cap = open_source(args)
    if args.input is None:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"[Sender] Streaming to {UDP_IP}:{UDP_PORT} ... Press Ctrl+C to stop.")
    
    frame_id = 0
    frames = []
    
    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                if args.input is not None:
                    break
                time.sleep(0.01)
                continue
                
            resized = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            if len(frames) == NUM_FRAMES:
                data = encode_clip(frames)
                frames.clear()
                for packet in packetize(data, frame_id, MAX_PAYLOAD):
                    sock.sendto(packet, (UDP_IP, UDP_PORT))
                    # Pace large clips so UDP receive buffers can drain.
                    time.sleep(0.0005)
                frame_id = (frame_id + 1) % (2 ** 32)

            # Cap sender framerate slightly to avoid saturating local network buffers
            time.sleep(0.03) 
            
    except KeyboardInterrupt:
        print("\n[Sender] Stopped.")
    finally:
        cap.release()
        sock.close()

if __name__ == "__main__":
    main()
