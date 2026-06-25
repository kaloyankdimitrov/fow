import socket
import collections
import cv2
import numpy as np
import torch
from transformers import VideoMAEConfig, VideoMAEForPreTraining, AutoImageProcessor
from transformers.utils import cached_file

# --- 1. CONFIGURATION & DEVICE SETUP ---
UDP_IP = "0.0.0.0"
UDP_PORT = 5005
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_FRAMES = 16
TUBELET_SIZE = 2
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE  # 14x14 grid

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")

# --- 2. AUTOMATIC WEIGHT REMAPPING FIX ---
def load_repatched_videomae(model_name="MCG-NJU/videomae-base"):
    print("Loading model and patching mismatched attention weights...")
    config = VideoMAEConfig.from_pretrained(model_name)
    model = VideoMAEForPreTraining(config)
    
    try:
        model_file = cached_file(model_name, "model.safetensors")
        from safetensors.torch import load_file
        state_dict = load_file(model_file)
    except Exception:
        model_file = cached_file(model_name, "pytorch_model.bin")
        state_dict = torch.load(model_file, map_location="cpu")
        
    fixed_state_dict = {}
    for k, v in state_dict.items():
        if "attention.attention.q_bias" in k:
            fixed_state_dict[k.replace("q_bias", "query.bias")] = v
            fixed_state_dict[k.replace("q_bias", "key.bias")] = torch.zeros_like(v)
        elif "attention.attention.v_bias" in k:
            fixed_state_dict[k.replace("v_bias", "value.bias")] = v
        else:
            fixed_state_dict[k] = v
            
    model.load_state_dict(fixed_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, AutoImageProcessor.from_pretrained(model_name)

model, processor = load_repatched_videomae()

# Pre-cache normalization tensors directly on GPU to bypass processor bottleneck
image_mean = torch.tensor(processor.image_mean, device=device).view(1, 1, 3, 1, 1)
image_std = torch.tensor(processor.image_std, device=device).view(1, 1, 3, 1, 1)

# --- 3. CUSTOM 3D UNPATCHIFY LOGIC ---
def unpatchify_videomae(logits, config):
    """Reverses full-sequence video tubelet patchification back to a readable video tensor."""
    batch_size = logits.shape[0]
    H_p = config.image_size // config.patch_size
    W_p = config.image_size // config.patch_size
    T_p = NUM_FRAMES // config.tubelet_size
    
    x = logits.view(batch_size, T_p, H_p, W_p, config.tubelet_size, config.patch_size, config.patch_size, config.num_channels)
    x = x.permute(0, 1, 4, 7, 2, 5, 3, 6).contiguous()
    video = x.view(batch_size, NUM_FRAMES, config.num_channels, config.image_size, config.image_size)
    return video

# --- 4. STREAMING HISTORY MANAGER ---
frame_history = collections.deque(maxlen=NUM_FRAMES)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"Receiver listening on UDP port {UDP_PORT}...")

try:
    while True:
        packet, addr = sock.recvfrom(65507)
        np_arr = np.frombuffer(packet, dtype=np.uint8)
        incoming_frame = cv2.cvtColor(cv2.imdecode(np_arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        
        if incoming_frame is None:
            continue
            
        incoming_frame = cv2.resize(incoming_frame, (IMAGE_SIZE, IMAGE_SIZE))
        
        # 1. Fast Vectorized Mask Detection (Accounts for JPEG Noise)
        patches = incoming_frame.reshape(GRID_SIZE, PATCH_SIZE, GRID_SIZE, PATCH_SIZE, 3).transpose(0, 2, 1, 3, 4)
        patch_means = patches.mean(axis=(2, 3, 4))
        mask_detected = patch_means < 15.0  # Safe threshold for compression artifacts
        
        # Cold start handling
        if len(frame_history) < NUM_FRAMES:
            while len(frame_history) < NUM_FRAMES:
                frame_history.append(incoming_frame.copy())
        else:
            frame_history.append(incoming_frame.copy())

        # Build spatiotemporal mask coordinates
        T_p = NUM_FRAMES // TUBELET_SIZE
        mask_grid = np.zeros((T_p, GRID_SIZE, GRID_SIZE), dtype=bool)
        mask_grid[-1] = mask_detected  # Target only the newest frame tubelet segment

        bool_masked_pos = torch.from_numpy(mask_grid.flatten()).unsqueeze(0).to(device)

        # 2. High-Speed Vectorized GPU Preprocessing (Bypasses HF Processor)
        history_np = np.stack(frame_history, axis=0)  # Shape: (16, 224, 224, 3)
        history_tensor = torch.from_numpy(history_np).to(device, dtype=torch.float32)
        # REMOVED: history_tensor = history_tensor[..., [2, 1, 0]] (Frame is already RGB)
        
        # Scale and Normalize on GPU
        pixel_values = history_tensor.permute(0, 3, 1, 2).unsqueeze(0) / 255.0
        pixel_values = (pixel_values - image_mean) / image_std

        with torch.inference_mode():  # Optimized context manager for raw inference speed
            outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
            
            batch_size = 1
            patch_dim = TUBELET_SIZE * PATCH_SIZE * PATCH_SIZE * 3  # 1536
            total_patches = T_p * GRID_SIZE * GRID_SIZE             # 1568
            
            full_logits = torch.zeros((batch_size, total_patches, patch_dim), device=device)
            
            if torch.any(bool_masked_pos):
                full_logits[bool_masked_pos] = outputs.logits.view(-1, patch_dim)
            
            reconstructed_video = unpatchify_videomae(full_logits, model.config)
            
        # Denormalize pixels back to RGB space
        reconstructed_video = (reconstructed_video * image_std + image_mean).clamp(0.0, 1.0)
        
        # Pull out the processed final frame (Index 15) as RGB uint8
        recon_frame_rgb = (reconstructed_video[0, -1].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        # 3. Vectorized Composite Blending (Replaces Nested Loops)
        mask_pixel = np.repeat(np.repeat(mask_detected, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1)
        final_output = incoming_frame.copy()
        final_output[mask_pixel] = recon_frame_rgb[mask_pixel]  # FIXED: Blending RGB into RGB

        # Feed the fully repaired frame back into memory history for the next iteration loop
        frame_history[-1] = final_output

        cv2.imshow("Restored Video Feed (VideoMAE Spatiotemporal)", cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    sock.close()
    cv2.destroyAllWindows()