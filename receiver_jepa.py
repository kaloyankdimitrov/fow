import torch
import torch.nn as nn
from einops import rearrange

class VideoPatchEmbedding(nn.Module):
    """Chunks video into Spacetime Patches."""
    def __init__(self, patch_size=(2, 16, 16), in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        # 3D convolution handles space and time simultaneously
        self.proj = nn.Conv3d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x shape: [B, C, T, H, W]
        x = self.proj(x) # [B, D, T_p, H_p, W_p]
        # Flatten spatio-temporal grids into a token sequence
        x = rearrange(x, 'b d t h w -> b (t h w) d')
        return x

class JEPAPredictor(nn.Module):
    """Predicts missing target tokens from context tokens."""
    def __init__(self, embed_dim=768, num_heads=8, depth=4):
        super().__init__()
        # Positional encodings for full patch sequences
        self.predictor_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, 
                dim_feedforward=embed_dim*4, batch_first=True
            ) for _ in range(depth)
        ])
        
    def forward(self, context_tokens, mask_tokens):
        # Merge context tokens and trainable mask tokens based on coordinates
        # For simplicity, we concatenate them here representing the full sequence
        x = torch.cat([context_tokens, mask_tokens], dim=1)
        for block in self.predictor_blocks:
            x = block(x)
        return x

class PixelDecoder(nn.Module):
    """Maps latent token representations back to raw pixel values."""
    def __init__(self, embed_dim=768, patch_size=(2, 16, 16), out_channels=3):
        super().__init__()
        self.patch_size = patch_size
        p_t, p_h, p_w = patch_size
        # Direct linear projection to total flattened patch size
        self.linear = nn.Linear(embed_dim, out_channels * p_t * p_h * p_w)

    def forward(self, latent_tokens, original_grid_shape):
        # latent_tokens: [B, N, D]
        B, N, _ = latent_tokens.shape
        p_t, p_h, p_w = self.patch_size
        
        # Project tokens back to pixels
        pixels = self.linear(latent_tokens) # [B, N, C * p_t * p_h * p_w]
        
        # Reshape to 3D video patches
        pixels = rearrange(
            pixels, 'b (t h w) (c pt ph pw) -> b c (t pt) (h ph) (w pw)',
            t=original_grid_shape[0], h=original_grid_shape[1], w=original_grid_shape[2],
            pt=p_t, ph=p_h, pw=p_w, c=3
        )
        return pixels

class VideoJEPARestorer(nn.Module):
    """Complete Framework uniting JEPA and Pixel Decoding."""
    def __init__(self, embed_dim=768):
        super().__init__()
        self.patch_embed = VideoPatchEmbedding(embed_dim=embed_dim)
        
        # JEPA Backbone Context Encoder
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12, batch_first=True),
            num_layers=6
        )
        
        # JEPA Latent Predictor
        self.predictor = JEPAPredictor(embed_dim=embed_dim)
        
        # Learnable prototype token for masked regions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Downstream Decoder
        self.decoder = PixelDecoder(embed_dim=embed_dim)

    def forward(self, video, patch_mask):
        # video: [B, C, T, H, W]
        # patch_mask: [B, N] boolean mask (True = corrupted/missing, False = visible)
        
        # 1. Patchify Video
        tokens = self.patch_embed(video) # [B, N, D]
        B, N, D = tokens.shape
        
        # Calculate patch grid dimensions for reconstruction math
        t_p = video.shape[2] // 2
        h_p = video.shape[3] // 16
        w_p = video.shape[4] // 16
        grid_shape = (t_p, h_p, w_p)
        
        # 2. Extract context (visible) tokens
        # For simplicity, we show a masked conditional pass
        context_mask = ~patch_mask
        
        # JEPA training typically processes only visible tokens through the encoder
        # Here we simulate token filtering (assuming uniform masking for demo)
        visible_tokens = tokens[context_mask].view(B, -1, D)
        
        # 3. Encode context
        context_latents = self.context_encoder(visible_tokens)
        
        # 4. Prepare Mask Tokens for prediction
        num_masked = patch_mask.sum(dim=1)[0].item()
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        
        # 5. Predict missing representation vectors in Latent Space
        predicted_latents = self.predictor(context_latents, mask_tokens)
        
        # 6. Decode back to original video pixels
        # For evaluation, we pass all tokens back to get a full sequence reconstruction
        reconstructed_video = self.decoder(predicted_latents, grid_shape)
        
        return reconstructed_video

import socket
import cv2
import numpy as np
import torch
from collections import deque

# Ensure model definitions are imported or defined above this wrapper script
# from your_model_file import VideoJEPARestorer

# --- Configuration (Must match sender) ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
PATCH_SIZE = 16
GRID_SIZE = 14
TOTAL_PATCHES_PER_FRAME = GRID_SIZE * GRID_SIZE # 196

# --- Model & Processing Constraints ---
TIME_FRAMES = 4  # The 3D-Conv layer expects T=4 frames
BUFFER_SIZE = 65535  # Max UDP packet size

def preprocess_frame(cv_img):
    """Convert openCV BGR image to PyTorch tensor normalized to [0, 1]."""
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() # [C, H, W]
    return img_tensor / 255.0

def compute_patch_mask(cv_img):
    """
    Scans the received image to find which patches are zeroed-out.
    Returns a 1D boolean array where True = Missing/Masked, False = Visible.
    """
    mask = np.zeros(TOTAL_PATCHES_PER_FRAME, dtype=bool)
    for idx in range(TOTAL_PATCHES_PER_FRAME):
        i, j = divmod(idx, GRID_SIZE)
        y1, y2 = i * PATCH_SIZE, (i + 1) * PATCH_SIZE
        x1, x2 = j * PATCH_SIZE, (j + 1) * PATCH_SIZE
        
        # Check if the entire patch consists of black pixels (zeros)
        patch = cv_img[y1:y2, x1:x2]
        if np.all(patch == 0):
            mask[idx] = True # Mark as missing
    return mask

def main():
    # 1. Initialize Network (Move to GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the restorer from your previous architectural layout
    model = VideoJEPARestorer().to(device)
    model.eval() # Inference mode
    
    # 2. Setup UDP Socket Network
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Receiver listening on {UDP_IP}:{UDP_PORT}... Waiting for sender.")

    # 3. Temporal Frame Deque for 3D Video Windows
    frame_buffer = deque(maxlen=TIME_FRAMES)
    mask_buffer = deque(maxlen=TIME_FRAMES)

    try:
        while True:
            # Receive JPEG stream chunks from socket
            data, _ = sock.recvfrom(BUFFER_SIZE)
            if not data:
                continue

            # Decode byte payload back to OpenCV image canvas
            np_arr = np.frombuffer(data, dtype=np.uint8)
            masked_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if masked_frame is None:
                continue

            # Analyze the frame structure to discover the missing patch sequence
            frame_mask = compute_patch_mask(masked_frame)
            
            # Preprocess and append tensors into temporal sliding buffer windows
            tensor_frame = preprocess_frame(masked_frame)
            frame_buffer.append(tensor_frame)
            mask_buffer.append(torch.from_numpy(frame_mask))

            # Trigger inference only when a complete 4-frame window exists
            if len(frame_buffer) == TIME_FRAMES:
                # Construct Video Tensor: [C, T, H, W] -> Add batch dim: [1, C, T, H, W]
                video_tensor = torch.stack(list(frame_buffer), dim=1).unsqueeze(0).to(device)
                
                # Construct Sequential Spacetime Mask: Flattened [1, N_total]
                # N_total = T_patches * H_patches * W_patches -> (4/2) * 14 * 14 = 392 tokens
                video_mask = torch.stack(list(mask_buffer), dim=0) # [T_frames, Patches_per_frame]
                
                # Reshape temporal mask sequences into token dimension sequences matching the Conv3D output 
                # Our 3D block processes T in groups of 2. So we group adjacent frames together.
                video_mask = video_mask.view(TIME_FRAMES // 2, 2, TOTAL_PATCHES_PER_FRAME)
                video_mask = video_mask.any(dim=1).view(1, -1).to(device) # [1, Total_Tokens]

                with torch.no_grad():
                    # Execute Forward Inference 
                    restored_video = model(video_tensor, video_mask)
                    
                    # Pull out the most recent reconstructed frame from the batch sequence
                    # restored_video shape is [B, C, T, H, W] -> target last frame index
                    out_frame = restored_video[0, :, -1, :, :].cpu().clamp(0, 1)
                    out_frame = (out_frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)

                # Render outputs side by side
                visual_comparison = np.hstack((masked_frame, out_frame))
                cv2.imshow("Receiver Stream: Masked vs Restored Output", visual_comparison)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
