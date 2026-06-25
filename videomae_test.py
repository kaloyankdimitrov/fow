import collections
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, VideoMAEForPreTraining

def main():
    # 1. Setup device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Correct loading of weights and configuration
    model_name = "MCG-NJU/videomae-base"
    print(f"Loading pre-trained weights for {model_name}...")
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = VideoMAEForPreTraining.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Model constants based on architecture configuration
    image_size = model.config.image_size        # 224
    patch_size = model.config.patch_size        # 16
    tubelet_size = model.config.tubelet_size    # 2
    num_frames = 16                             # VideoMAE standard clip length
    num_channels = 3

    # Calculate spatial-temporal patching properties
    num_patches_per_frame = (image_size // patch_size) ** 2
    seq_length = (num_frames // tubelet_size) * num_patches_per_frame
    flat_patch_dim = tubelet_size * (patch_size ** 2) * num_channels

    # ImageNet normalization constants used by the processor (needed for reverse processing)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1).to(device)

    # 3. Setup OpenCV Webcam Stream
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam stream.")
        return

    print("Webcam stream running. Press 'q' to exit.")
    
    # Use a rolling queue buffer to hold the 16 context frames
    frame_buffer = collections.deque(maxlen=num_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process individual frame for the buffer queue
        # Resize to matching model input size and change BGR (OpenCV) to RGB
        resized_frame = cv2.resize(frame, (image_size, image_size))
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        frame_buffer.append(rgb_frame)

        # Proceed only when we have enough temporal context frames (16)
        if len(frame_buffer) == num_frames:
            # Convert list of frames into normalized pixel values tensor
            inputs = image_processor(list(frame_buffer), return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device) # Shape: (1, 16, 3, 224, 224)

            # 4. Generate a masking topology (e.g., 80% masking ratio)
            mask_ratio = 0.80
            rand_indices = torch.rand(1, seq_length)
            bool_masked_pos = (rand_indices < mask_ratio).to(device, dtype=torch.bool)

            with torch.no_grad():
                # Model outputs logits matching only the hidden/masked patches
                outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
                logits = outputs.logits # Shape: (1, num_masked_patches, flat_patch_dim)

            # 5. Correct Processing & Un-patchification of Pixel Values
            # Break down the original full tensor into flat structural patches
            B, T, C, H, W = pixel_values.shape
            cubes = pixel_values.view(B, T // tubelet_size, tubelet_size, C, H // patch_size, patch_size, W // patch_size, patch_size)
            cubes = cubes.permute(0, 1, 4, 6, 2, 5, 7, 3) # Match Transformer tracking order
            orig_sequence = cubes.flatten(1, 3).flatten(2) # Shape: (1, seq_length, flat_patch_dim)

            # Reconstruct the sequence by injecting predicted patches over masked positions
            filled_sequence = orig_sequence.clone()
            filled_sequence[bool_masked_pos] = logits.reshape(-1, flat_patch_dim)

            # To visualize what the model strictly received, create a masked view
            masked_sequence = orig_sequence.clone()
            masked_sequence[bool_masked_pos] = -2.0 # Dark placeholder value for normalized space

            # Transform sequence blocks back to continuous 3D video tensors
            def sequence_to_video(seq):
                recon_cubes = seq.view(B, T // tubelet_size, H // patch_size, W // patch_size, tubelet_size, patch_size, patch_size, C)
                recon_cubes = recon_cubes.permute(0, 1, 4, 7, 2, 5, 3, 6)
                video_tensor = recon_cubes.reshape(B, T, C, H, W)
                # Reverse ImageNet Normalization: (Pixel * std) + mean
                video_tensor = video_tensor * std + mean
                return torch.clamp(video_tensor, 0.0, 1.0)

            restored_video = sequence_to_video(filled_sequence)
            masked_video = sequence_to_video(masked_sequence)

            # Extract the most recent frame (-1) from the batch for real-time streaming feedback
            def tensor_to_cv2(tensor_clip):
                frame_np = (tensor_clip[0, -1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                return cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

            cv2_masked = tensor_to_cv2(masked_video)
            cv2_restored = tensor_to_cv2(restored_video)
            cv2_original = cv2.resize(frame, (image_size, image_size))

            # Display panels side by side: Original Context | Corrupted/Masked Input | VideoMAE Restored
            display_panel = np.hstack((cv2_original, cv2_masked, cv2_restored))
            cv2.imshow("VideoMAE Live Restoration (Original | Masked Input | Restored)", display_panel)
        else:
            # Display fallback initialization text while filling the buffer
            cv2.putText(frame, "Buffering frames...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("VideoMAE Live Restoration (Original | Masked Input | Restored)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()