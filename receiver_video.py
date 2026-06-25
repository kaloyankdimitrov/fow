import socket
import collections
import cv2
import numpy as np
import torch
from transformers import VideoMAEForPreTraining, AutoImageProcessor

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

# --- 2. CLEAN MODEL LOADING ---
def load_repatched_videomae(model_name="MCG-NJU/videomae-base"):
    print("Patching attention weight naming drift...")
    from transformers import VideoMAEConfig, VideoMAEForPreTraining
    from transformers.utils import cached_file
    from safetensors.torch import load_file

    config = VideoMAEConfig.from_pretrained(model_name)
    model = VideoMAEForPreTraining(config)
    
    try:
        model_file = cached_file(model_name, "model.safetensors")
        state_dict = load_file(model_file)
    except Exception:
        model_file = cached_file(model_name, "pytorch_model.bin")
        state_dict = torch.load(model_file, map_location="cpu")
        
    fixed_state_dict = {}
    for k, v in state_dict.items():
        if "attention.attention.q_bias" in k:
            fixed_state_dict[k.replace("q_bias", "query.bias")] = v
            # The original architecture does not utilize key bias; fill with zeros to satisfy HF architecture
            fixed_state_dict[k.replace("q_bias", "key.bias")] = torch.zeros_like(v)
        elif "attention.attention.v_bias" in k:
            fixed_state_dict[k.replace("v_bias", "value.bias")] = v
        else:
            fixed_state_dict[k] = v
            
    # strict=True confirms every single parameter successfully aligned
    model.load_state_dict(fixed_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, AutoImageProcessor.from_pretrained(model_name)

model, processor = load_repatched_videomae()

# Pre-cache normalization tensors directly on GPU
image_mean = torch.tensor(processor.image_mean, device=device).view(1, 1, 3, 1, 1)
image_std = torch.tensor(processor.image_std, device=device).view(1, 1, 3, 1, 1)

def patchify_videomae(pixel_values, config):
    """VideoMAE patchify equivalent. Input: (B, T, C, H, W) -> Output: (B, num_patches, tubelet*patch*patch*3)"""
    B, T, C, H, W = pixel_values.shape
    p, u = config.patch_size, config.tubelet_size
    h, w, t = H // p, W // p, T // u

    x = pixel_values.reshape(B, t, u, C, h, p, w, p)
    x = x.permute(0, 1, 4, 6, 2, 5, 7, 3)
    patches = x.reshape(B, t * h * w, u * p * p * C)
    return patches

def unpatchify_videomae(logits, config):
    """Reverses full-sequence video tubelet patchification back to a readable [0, 1] video tensor."""
    batch_size = logits.shape[0]
    H_p = W_p = config.image_size // config.patch_size
    T_p = NUM_FRAMES // config.tubelet_size
    
    x = logits.view(batch_size, T_p, H_p, W_p, config.tubelet_size, config.patch_size, config.patch_size, config.num_channels)
    x = x.permute(0, 1, 4, 7, 2, 5, 3, 6).contiguous()
    video = x.view(batch_size, NUM_FRAMES, config.num_channels, config.image_size, config.image_size)
    return video

# --- 3. STREAMING HISTORY MANAGER ---
frame_history = collections.deque(maxlen=NUM_FRAMES)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"Receiver listening on UDP port {UDP_PORT}...")

try:
    while True:
        packet, addr = sock.recvfrom(65507)
        np_arr = np.frombuffer(packet, dtype=np.uint8)
        decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if decoded is None:
            continue
            
        incoming_frame = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        incoming_frame = cv2.resize(incoming_frame, (IMAGE_SIZE, IMAGE_SIZE))
        
        # 1. Fast Vectorized Mask Detection
        patches = incoming_frame.reshape(GRID_SIZE, PATCH_SIZE, GRID_SIZE, PATCH_SIZE, 3).transpose(0, 2, 1, 3, 4)
        patch_means = patches.mean(axis=(2, 3, 4))
        mask_detected = patch_means < 15.0  
        
        if len(frame_history) < NUM_FRAMES:
            while len(frame_history) < NUM_FRAMES:
                frame_history.append(incoming_frame.copy())
        else:
            frame_history.append(incoming_frame.copy())

        T_p = NUM_FRAMES // TUBELET_SIZE
        mask_grid = np.zeros((T_p, GRID_SIZE, GRID_SIZE), dtype=bool)
        mask_grid[-1] = mask_detected  

        bool_masked_pos = torch.from_numpy(mask_grid.flatten()).unsqueeze(0).to(device)

        # 2. High-Speed Vectorized GPU Preprocessing
        history_np = np.stack(frame_history, axis=0)  
        history_tensor = torch.from_numpy(history_np).to(device, dtype=torch.float32)
        
        # Obtain clean base video in [0, 1] range
        raw_video = history_tensor.permute(0, 3, 1, 2).unsqueeze(0) / 255.0
        
        # Compute input normalized specifically for the Model's Encoder
        pixel_values = (raw_video - image_mean) / image_std

        with torch.inference_mode():
            outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)

            # --- FIX 1 & 2: Create a mask-free reference in raw [0, 1] space for statistics ---
            clean_reference_video = raw_video.clone()
            # Replace the black-masked last frame with the fully repaired second-to-last frame
            clean_reference_video[0, -1] = clean_reference_video[0, -2]
            
            # Patchify the clean reference video
            clean_patches = patchify_videomae(clean_reference_video, model.config)

            if model.config.norm_pix_loss:
                patch_mean = clean_patches.mean(dim=-1, keepdim=True)
                # VideoMAE uses unbiased variance matching training
                patch_var = clean_patches.var(dim=-1, keepdim=True, unbiased=True)

                # Denormalize the model predictions straight into clean [0, 1] space
                pred_rgb = (
                    outputs.logits *
                    torch.sqrt(patch_var[bool_masked_pos] + 1e-6)
                    +
                    patch_mean[bool_masked_pos]
                )
            else:
                pred_rgb = outputs.logits

            # --- FIX 3: Reconstruct the full patch sequence entirely within [0, 1] space ---
            full_patches_unnorm = patchify_videomae(raw_video, model.config)
            reconstructed_patches = full_patches_unnorm.clone()

            if torch.any(bool_masked_pos):
                reconstructed_patches[bool_masked_pos] = pred_rgb

            # Unpatchify directly into final image format (No global denormalization loop needed)
            reconstructed_video = unpatchify_videomae(reconstructed_patches, model.config).clamp(0.0, 1.0)
            
        # 3. Vectorized Composite Blending
        mask_pixel = np.repeat(np.repeat(mask_detected, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1)
        recon_frame_rgb = (reconstructed_video[0, -1].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

        final_output = incoming_frame.copy()
        alpha = 0.85

        final_output[mask_pixel] = (
            alpha * recon_frame_rgb[mask_pixel] +
            (1.0 - alpha) * incoming_frame[mask_pixel]
        ).astype(np.uint8)
        
        frame_history[-1] = final_output

        cv2.imshow("Restored Video Feed (VideoMAE Spatiotemporal)", cv2.cvtColor(final_output, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    sock.close()
    cv2.destroyAllWindows()