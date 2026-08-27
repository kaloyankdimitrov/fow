import cv2
import numpy as np
import socket
import onnxruntime as ort
from huggingface_hub import hf_hub_download

# --- Configuration ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
PATCH_SIZE = 16
GRID_SIZE = 14
MASK_RATIO = 0.75

print("Loading State-of-the-Art Inpainting Engine via ONNX Runtime...")

# Download model weights from the hub
model_path = hf_hub_download(repo_id="opencv/inpainting_lama", filename="inpainting_lama_2025jan.onnx")

# Set up optimized ONNX Runtime session (Bypasses OpenCV 4.x graph parsing limits)
available_providers = ort.get_available_providers()
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)

# Network configuration
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

print(f"Receiver engine active using: {session.get_providers()[0]}")
print("Listening for incoming sparse payloads...")

try:
    while True:
        try:
            packet, addr = sock.recvfrom(65535)
        except OSError:
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue
            
        if len(packet) < 4: 
            continue
            
        # Parse frame ID and image payload
        frame_id = int.from_bytes(packet[:4], byteorder='big')
        jpeg_bytes = np.frombuffer(packet[4:], dtype=np.uint8)
        
        masked_frame = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
        if masked_frame is None: 
            continue
            
        # Convert BGR to RGB to entirely eliminate the blue tint!
        frame_rgb = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        
        # Re-generate the identical binary mask locally using the frame_id seed
        np.random.seed(frame_id)
        total_patches = GRID_SIZE * GRID_SIZE
        num_keep = int(total_patches * (1 - MASK_RATIO))
        shuffled_indices = np.random.permutation(total_patches)
        keep_indices = shuffled_indices[:num_keep]
        
        # Build mathematical mask (255 = region to heal, 0 = keep region)
        mask = np.full((224, 224), 255, dtype=np.uint8)
        for idx in keep_indices:
            i, j = divmod(idx, GRID_SIZE)
            y1, y2 = i * PATCH_SIZE, (i + 1) * PATCH_SIZE
            x1, x2 = j * PATCH_SIZE, (j + 1) * PATCH_SIZE
            mask[y1:y2, x1:x2] = 0

        # Resize tensors to standard 512x512 model target dimensions
        frame_resized = cv2.resize(frame_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
            
        # Format tensors to standard NCHW dimensions expected by LaMa [1, 3, 512, 512]
        img_blob = frame_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_blob = np.expand_dims(img_blob, axis=0)
        
        mask_blob = mask_resized.astype(np.float32) / 255.0
        mask_blob = np.expand_dims(mask_blob, axis=(0, 1)) # Format to [1, 1, 512, 512]
        mask_blob = (mask_blob > 0).astype(np.float32)
        
        # Execute ONNX Runtime Forward Pass
        inputs = {
            session.get_inputs()[0].name: img_blob,
            session.get_inputs()[1].name: mask_blob
        }
        outputs = session.run(None, inputs)
        output = outputs[0]
        
        # Post-process tensor to an image frame array
        output = np.squeeze(output, axis=0).transpose(1, 2, 0)
        
        # Directly clip to safe 8-bit image boundaries and cast
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        # Scale back down to 224x224 and overlay seamlessly over the received frame
        output_224 = cv2.resize(output, (224, 224), interpolation=cv2.INTER_LINEAR)
        output_bgr = cv2.cvtColor(output_224, cv2.COLOR_RGB2BGR)
        
        mask_expanded = np.expand_dims(mask, axis=2) / 255.0
        seamless_frame = (output_bgr * mask_expanded + masked_frame * (1 - mask_expanded)).astype(np.uint8)        
        cv2.imshow("Restored SOTA Output Feed", seamless_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
finally:
    cv2.destroyAllWindows()