# FOW · Fog of War

**Send part of the picture. Let the model fill in the fog.**

FOW is a playground for partial video transmission and generative reconstruction. It explores how much of a camera stream can be omitted, what information still needs to travel over the network, and how different models rebuild the missing content.

The experiments range from image masked autoencoders and learned latent transmission to LaMa inpainting and VideoMAE reconstruction across time. Each is a small Python prototype with a live OpenCV display.

```text
Camera or video file → Select pixels or encode tokens → UDP → Reconstruct → Display
                  Sender                         Receiver
```

The reconstructed video is a prediction of missing content, not an exact recovery of the original. Mask percentages describe omitted patches or tokens; they are not measured bandwidth savings. Metadata, keyframes, compression, and transport overhead all affect the bytes sent.

## Experiments at a glance

| Experiment | Sender → Receiver | Model | What travels | Default omission |
| --- | --- | --- | --- | --- |
| Sparse image patches | [`sender.py`](sender.py) → [`receiver.py`](receiver.py) | ViT-MAE Large | Visible pixel patches and their positions | 75% of spatial patches |
| Learned latents | [`sender_latent.py`](sender_latent.py) → [`receiver_latent.py`](receiver_latent.py) | ViT-MAE Base | Encoder latents, restoration map, and patch statistics | 75% of encoder patch tokens |
| Seeded-mask inpainting | [`sender_inpainting.py`](sender_inpainting.py) → [`receiver_inpainting.py`](receiver_inpainting.py) | LaMa via ONNX Runtime | Sparse JPEG and a frame ID used to reproduce the mask | 75% of spatial patches |
| Sparse JPEG with video context | [`sender_video.py`](sender_video.py) → [`receiver_video.py`](receiver_video.py) | VideoMAE Base | Sparse JPEG frames; mask inferred at receiver | 25% of spatial patches |
| Masked video clips | [`sender_mae.py`](sender_mae.py) → [`receiver_mae.py`](receiver_mae.py) | VideoMAE Base | Visible tubelets, explicit mask, and reconstruction statistics | 75% of space-time tokens |
| Keyframes and sparse updates | [`lama_sender.py`](lama_sender.py) → [`lama_receiver.py`](lama_receiver.py) | LaMa via SimpleLama | Full JPEG keyframes plus sparse raw-pixel updates | 90% of patches on delta frames |
| Learned video compression | [`sender_dcvc.py`](sender_dcvc.py) → [`receiver_dcvc.py`](receiver_dcvc.py) | DCVC-RT | Entropy-coded I/P frames over TCP | Quantization-controlled compression |

These are separate wire protocols. Use the matching sender and receiver in each row.

## 1. Sparse image patches · ViT-MAE

**Question:** Can a single-frame masked autoencoder reconstruct a scene from a quarter of its patches?

The sender center-crops the camera image to 224 × 224 and divides it into a 14 × 14 grid of 16 × 16 patches. It randomly keeps 49 of the 196 patches, then sends their pixels and indices in a pickled UDP payload.

The receiver rebuilds the sparse canvas and uses `facebook/vit-mae-large`, passing a custom noise tensor to guide the model's masking order. Its display places received patches beside the reconstruction. This is the image-only baseline, without temporal context.

**Implementation note:** `receiver.py` explicitly loads onto Apple MPS. Other devices require changing its model-loading configuration.

<!--
> **GIF demo — `receiver.py`**
>
> Received patches → ViT-MAE reconstruction
>
> Add a recording at `docs/demos/receiver.gif` and uncomment the image below.
-->

<!-- ![ViT-MAE sparse image reconstruction](docs/demos/receiver.gif) -->

## 2. Learned latent transmission · ViT-MAE

**Question:** What changes when the sender transmits encoder features instead of visible pixels?

Here the sender runs the `facebook/vit-mae-base` encoder locally. Its default masking drops 75% of image patch tokens. The payload contains the remaining encoded representation, the token restoration map, and per-patch mean and standard deviation values.

Pickled payloads are split into 1,000-byte chunks with frame and chunk identifiers. The receiver assembles a frame, runs the MAE decoder, applies the transmitted statistics, and renders the resulting 224 × 224 reconstruction.

This experiment moves model computation onto both ends of the link. Its unquantized tensor payload should not be assumed smaller than a compressed image simply because fewer tokens are retained.

<!--
> **GIF demo — `receiver_latent.py`**
>
> Video reconstructed from transmitted encoder latents
>
> Add a recording at `docs/demos/receiver_latent.gif` and uncomment the image below.
-->

<!-- ![ViT-MAE latent stream reconstruction](docs/demos/receiver_latent.gif) -->

## 3. Seeded-mask image inpainting · LaMa ONNX

**Question:** Can an image inpainting model restore a sparse stream without receiving an explicit mask?

The sender keeps 25% of the 16 × 16 patches in a 224 × 224 image and blacks out the rest. It sends the sparse canvas as a quality-85 JPEG prefixed with a four-byte frame ID. That ID seeds the patch selection on both sides, allowing the receiver to reproduce the mask.

The receiver loads `opencv/inpainting_lama` / `inpainting_lama_2025jan.onnx` through ONNX Runtime. It resizes the image and mask to 512 × 512, performs inpainting, then scales the result back and composites predicted regions with received pixels. It selects the CUDA execution provider when available, otherwise CPU.

<!--
> **GIF demo — `receiver_inpainting.py`**
>
> LaMa reconstruction of a stream with 75% of patches hidden
>
> Add a recording at `docs/demos/receiver_inpainting.gif` and uncomment the image below.
-->

<!-- ![LaMa ONNX sparse video inpainting](docs/demos/receiver_inpainting.gif) -->

## 4. Sparse JPEG with temporal context · VideoMAE

**Question:** Can recent reconstructed frames help fill gaps in the next sparse frame?

The sender transmits 224 × 224 JPEG frames with randomly selected patches blacked out. Its current `MASK_RATIO = 0.25` hides **25%** of patches, despite the adjacent comment saying 75%.

The receiver uses `MCG-NJU/videomae-base` and a rolling history of 16 frames. It detects missing patches using a brightness threshold, marks them in the last temporal token group, and estimates reconstruction statistics from recent context. It preserves received pixels, fills missing regions with predictions, and feeds the restored frame back into history. Per-channel reconstruction statistics are estimated from repaired history, with spatial inpainting used to bootstrap the initial context.

The receiver uses Transformers checkpoint loading and runs inference in a background thread while the main loop drains pending JPEG packets and updates the display. The brightness-based mask can confuse naturally dark image regions with omitted data, and feeding predictions back into history can accumulate errors.

<!--
> **GIF demo — `receiver_video.py`**
>
> VideoMAE restoration using sparse input and reconstructed history
>
> Add a recording at `docs/demos/receiver_video.gif` and uncomment the image below.
-->

<!-- ![VideoMAE sparse JPEG temporal reconstruction](docs/demos/receiver_video.gif) -->

## 5. Sender-masked video clips · VideoMAE

**Question:** What happens when masking is defined explicitly across a complete video clip?

The sender collects 16 RGB frames at 224 × 224. Each clip contains 1,568 space-time tokens: eight temporal groups, each with a 14 × 14 spatial grid. A token covers two frames and a 16 × 16 spatial patch.

Masking happens on the sender. It randomly removes 75% of tokens and transmits:

- The 392 visible tubelets as raw pixel data.
- A packed binary mask identifying the missing tokens.
- Per-channel mean and standard deviation metadata for masked tubelets, used to restore the scale of normalized predictions.

[`mae_protocol.py`](mae_protocol.py) defines the versioned `MAE1` payload, zlib compression, and UDP fragmentation into 1,400-byte payload chunks. Masked pixel arrays are omitted; their summary statistics are still transmitted.

The receiver runs `MCG-NJU/videomae-base` with the sender's mask, preserves visible pixels, and fills missing tubelets with predictions. Network reception and inference run in separate background threads. The display shows the last masked frame beside its reconstruction, with packet, clip, and error counters.

This is a clip-based experiment: collecting 16 frames adds latency, and the display shows the final frame of each reconstructed clip rather than playing all 16 frames.

<!--
> **GIF demo — `receiver_mae.py`**
>
> Sender-masked input → VideoMAE reconstruction
>
> Add a recording at `docs/demos/receiver_mae.gif` and uncomment the image below.
-->

<!-- ![VideoMAE sender-masked clip reconstruction](docs/demos/receiver_mae.gif) -->

## 6. Keyframes and sparse updates · LaMa

**Question:** Can occasional full frames anchor reconstruction while intermediate frames carry only a few updates?

At 320 × 240, the sender divides each frame into 300 patches of 16 × 16 pixels. Every 30 frames it sends a full quality-60 JPEG keyframe in 1,000-byte chunks. Between keyframes, it selects 30 patches—10% of the frame—and sends their raw pixels and coordinates in groups of up to four patches per packet.

The receiver uses `SimpleLama`. A decoded keyframe establishes the reference image. For each complete delta frame, it overlays received patches on that reference, marks all other areas for inpainting, and uses the reconstructed output as the next reference. Incomplete keyframe and delta assemblies expire after 0.5 seconds when cleanup runs on incoming packets.

The receiver exposes three views: the sparse/reference canvas, the missing-region mask, and the reconstructed frame. Full keyframes periodically refresh the reference, while intervening predictions can drift.

<!--
> **GIF demo — `lama_receiver.py`**
>
> Sparse updates + missing-region mask + LaMa reconstruction
>
> Add a recording at `docs/demos/lama_receiver.gif` and uncomment the image below.
-->

<!-- ![LaMa keyframe and sparse update reconstruction](docs/demos/lama_receiver.gif) -->

## 7. Learned video compression · DCVC-RT

> **Status: not tested end to end.** This experiment requires NVIDIA CUDA, which was unavailable in the development environment. Actual DCVC encoding, decoding, and live streaming have not been validated. 

[`sender_dcvc.py`](sender_dcvc.py) → [`receiver_dcvc.py`](receiver_dcvc.py), with the shared adapter in [`dcvc_common.py`](dcvc_common.py).

This experiment sends actual entropy-coded neural video bitstreams. The sender uses the DCVC-RT image codec for periodic I frames and its predictive video codec for intervening P frames. The receiver reconstructs each frame from the bitstream and previously decoded references. It provides a compression baseline alongside FOW's sparse-pixel experiments; no random pixel mask is applied.

Transport uses **TCP port 5006** to preserve delivery and ordering of reference-dependent frames. Messages include a versioned header, frame number, dimensions, frame type, quantizer, and compressed bytes. The receiver decodes every frame but can skip display updates. TCP backpressure can increase latency when decoding cannot keep up. Restart both processes for a new session.

Defaults: 320 × 240, camera 0, target capture rate 30 FPS, base QP 32, and an I frame every 32 frames. The upstream temporal QP schedule adjusts P-frame quantizers. Both terminals report frame sizes and codec times; sender size includes FOW framing, receiver size is codec payload only. These timings exclude model loading, and do not measure end-to-end latency or TCP/IP overhead.

### DCVC environment and weights

Use an NVIDIA CUDA machine for **both** processes. This adapter does not support CPU or Apple MPS. It targets the official **DCVC-RT** implementation under `DCVC-family/DCVC-RT`, not the newer DCVC-UF at the repository root. Follow the [official DCVC-RT setup and checkpoint instructions](https://github.com/microsoft/DCVC/tree/main/DCVC-family/DCVC-RT) for its Python/PyTorch/CUDA environment and dependencies.

```bash
git clone https://github.com/microsoft/DCVC.git ../DCVC
cd ../DCVC/DCVC-family/DCVC-RT
python -m pip install -r requirements.txt
python -m pip install ./src/cpp
python -m pip install ./src/layers/extensions/inference
python -m pip install opencv-python
```

Download the official `cvpr2025_image.pth.tar` and `cvpr2025_video.pth.tar` checkpoints using the upstream README's pretrained-model link. Place them in that checkout's `checkpoints/` directory. Use the same upstream revision and checkpoint files on sender and receiver. The commands above assume the matching CUDA-enabled PyTorch environment and build tools are already installed.

After the one-time setup, run each command in its own terminal—no arguments required:

```bash
python receiver_dcvc.py
```

```bash
python sender_dcvc.py
```

Both scripts default to the DCVC checkout beside FOW (`../DCVC/DCVC-family/DCVC-RT`) and load the two checkpoint filenames above from its `checkpoints/` directory. This default is resolved relative to the scripts, so it also works when launched from another working directory. Edit `DCVC_ROOT` in `dcvc_common.py` to configure a different shared installation location.

Optional overrides include `--camera 1`, `--host RECEIVER_IP`, `--qp 32`, `--input recording.mp4`, and `--headless`. Advanced settings remain available through `--help`. If you pass `--dcvc-root`, checkpoint defaults follow that directory automatically; `--model-i` and `--model-p` can override individual files. Press Q/Escape in a preview window or Ctrl+C to stop.

Full encoding/decoding and visual quality still need validation with the official checkpoints on CUDA hardware. 

## Local VideoMAE studies

These webcam demos explore reconstruction without UDP. They retain the full source clip locally, so their patch statistics are available without transmission.

| Script | Focus | Default masking | Display |
| --- | --- | --- | --- |
| [`videomae_test.py`](videomae_test.py) | Initial 16-frame reconstruction loop with randomly sampled masks and direct prediction compositing | Approximately 80% | Original / masked / restored |
| [`videomae_test2.py`](videomae_test2.py) | Fixed-count masking, per-patch scalar statistics, and background inference | 75% | Original / masked / reconstructed |
| [`videomae_test3.py`](videomae_test3.py) | Per-channel patch statistics for reconstruction scaling, with background inference | 50% | Original / masked / reconstructed |

All three use `MCG-NJU/videomae-base`. The second and third demos accept `--camera`, `--device`, and `--skip`:

```bash
python videomae_test.py
python videomae_test2.py --camera 0 --device cpu --skip 1
python videomae_test3.py --camera 0 --device cpu --skip 1
```

<!--
Optional recordings: `docs/demos/videomae_test.gif`, `docs/demos/videomae_test2.gif`, and `docs/demos/videomae_test3.gif`.
-->

## Running an experiment

### Environment

The original project notes record **Python 3.13.13** for the main experiments and **Python 3.10** for the SimpleLama demo. There is no pinned dependency manifest yet; package and model compatibility can vary between experiments.

Create an environment for the experiment you want to run. A starting set for the PyTorch/Transformers demos is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy opencv-python torch torchvision transformers pillow huggingface-hub safetensors accelerate
```

For the ONNX inpainting pair, add `onnxruntime`. For the SimpleLama pair, use a separate Python 3.10 environment and install `simple-lama-inpainting`, `numpy`, and `opencv-python`.

Model-loading code downloads weights on first use, so allow network access and time for initialization. OpenCV windows require a graphical desktop. Senders require camera access only when using a webcam.

### Two terminals, one matching pair

For the explicit sender-masked VideoMAE experiment:

```bash
# Terminal 1: receiver
python receiver_mae.py
```

```bash
# Terminal 2: sender
python sender_mae.py
```

For another experiment, substitute the matching filenames from the table above. The earlier network experiments default to **UDP port 5005** (DCVC uses TCP port 5006), so run one pair at a time. Their payload formats are incompatible.

### Use a pre-recorded video

Every sender accepts `--input PATH` to read a video file instead of a webcam. Start its matching receiver first, then run one of:

```bash
python sender.py --input recording.mp4
python sender_latent.py --input recording.mp4
python sender_inpainting.py --input recording.mp4
python sender_video.py --input recording.mp4
python sender_mae.py --input recording.mp4
python lama_sender.py --input recording.mp4
python sender_dcvc.py --input recording.mp4
```

Use a video format supported by your OpenCV installation; quote paths containing spaces. Relative paths are resolved from the current working directory. `--input` takes precedence over `--camera` and never falls back to a webcam if the file cannot be opened. Omit `--input` to use live capture.

Files play once, and senders exit at end of file (or a failed frame read). Audio is not transmitted. Frames follow each experiment's existing resizing, masking, encoding, and timing; playback is not synchronized to the file's original frame rate. DCVC uses its `--fps` target (default 30). The clip sender transmits only complete 16-frame clips, discarding any shorter final clip.

### Configuration

- **Destination:** senders default to `127.0.0.1`. For two machines, set the sender's `UDP_IP` to the receiver's address. Receivers bound to `127.0.0.1` must also be changed to a reachable interface or `0.0.0.0`.
- **Camera:** all senders accept `--camera INDEX`. The UDP senders default to index `1`; `sender.py` and `sender_video.py` fall back to `0` only when no source option is supplied. DCVC defaults to index `0`. The later local VideoMAE demos also expose `--camera`.
- **Device:** most PyTorch scripts select CUDA when available, otherwise CPU. `receiver.py` explicitly uses MPS; `receiver_video.py` checks MPS, then CUDA, then CPU.
- **Masking and timing:** edit the constants near the top of each script. Keep matching sender/receiver assumptions synchronized; the clip experiment shares its format through `mae_protocol.py`.
- **Exit:** most preview windows accept `q`; some also accept Escape. `sender_mae.py` uses Ctrl+C. The SimpleLama sender accepts Escape, while its receiver can be stopped with Ctrl+C.

### If the clip receiver is waiting

`receiver_mae.py` shows and periodically prints reception counters. Zero packets points to the sender, destination address, or network path. Packets without completed clips indicate incomplete assemblies; UDP does not retransmit missing chunks. A masked image with “Reconstructing...” means a clip arrived and inference is pending. Check terminal messages for model or payload errors, and use matching versions of both scripts and `mae_protocol.py`.

## Scope and validation

FOW contains exploratory implementations, not a benchmark suite or a production video codec. It does not yet report measured bitrate, reconstruction quality, frame rate, or end-to-end latency across experiments. Earlier prototypes also include model-version assumptions and different normalization strategies, so visual output needs to be evaluated per experiment.

These checks do not validate live-camera model quality. Several older protocols deserialize Python pickle payloads and should only be run with trusted senders on a trusted network.

<!--
## Demo gallery contributions

Each receiver section has an image placeholder ready to uncomment once a GIF exists. Use the suggested `docs/demos/` paths, and include the model, masking ratio, hardware, and any changed settings alongside each recording so comparisons remain useful.
-->
