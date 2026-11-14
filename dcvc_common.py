"""FOW framing and adapter for Microsoft's DCVC-RT (CVPR 2025).

Upstream API: microsoft/DCVC, DCVC-family/DCVC-RT/test_video.py.
The wire contains a JSON frame description followed by an entropy-coded bitstream.
"""
import argparse
import json
from pathlib import Path
import struct
import sys

HEADER = struct.Struct('!4sII')
MAGIC = b'FDC1'
MAX_STREAM = 32 * 1024 * 1024

# Shared defaults for both scripts; independent of the working directory.
DCVC_ROOT = Path(__file__).resolve().parent.parent / 'DCVC/DCVC-family/DCVC-RT'


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def add_codec_args(parser):
    parser.add_argument('--dcvc-root', default=DCVC_ROOT, type=Path,
                        help='Codec directory (default: ../DCVC/DCVC-family/DCVC-RT beside FOW)')
    parser.add_argument('--model-i', type=Path,
                        help='Override checkpoints/cvpr2025_image.pth.tar under the codec directory')
    parser.add_argument('--model-p', type=Path,
                        help='Override checkpoints/cvpr2025_video.pth.tar under the codec directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--port', type=int, default=5006)
    parser.add_argument('--headless', action='store_true')


def resolve_codec_paths(args):
    args.dcvc_root = args.dcvc_root.expanduser().resolve()
    for name, filename in (('model_i', 'cvpr2025_image.pth.tar'),
                           ('model_p', 'cvpr2025_video.pth.tar')):
        path = getattr(args, name) or args.dcvc_root / 'checkpoints' / filename
        setattr(args, name, path.expanduser().resolve())
    return args


def validate_metadata(meta):
    if not isinstance(meta, dict):
        raise ValueError('Expected frame metadata object')
    for name in ('frame', 'width', 'height', 'qp'):
        if type(meta.get(name)) is not int:
            raise ValueError(f'Invalid {name}')
    if meta['frame'] < 0 or not 1 <= meta['width'] <= 4096 or not 1 <= meta['height'] <= 4096:
        raise ValueError('Invalid frame number or dimensions')
    if meta.get('kind') not in ('I', 'P') or not 0 <= meta['qp'] <= 71:
        raise ValueError('Invalid frame type or quantizer')
    if meta['kind'] == 'I' and meta['qp'] > 63:
        raise ValueError('Invalid intra quantizer')


def send_frame(sock, meta, stream):
    validate_metadata(meta)
    if not 0 < len(stream) <= MAX_STREAM:
        raise ValueError('Invalid bitstream size')
    description = json.dumps(meta, separators=(',', ':')).encode('utf-8')
    sock.sendall(HEADER.pack(MAGIC, len(description), len(stream)) + description + stream)
    return HEADER.size + len(description) + len(stream)


def read_exact(sock, size, allow_eof=False):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            if allow_eof and not chunks:
                return None
            raise EOFError('Connection ended in the middle of a frame')
        chunks.extend(chunk)
    return bytes(chunks)


def receive_frame(sock):
    header = read_exact(sock, HEADER.size, allow_eof=True)
    if header is None:
        return None
    magic, metadata_size, stream_size = HEADER.unpack(header)
    if magic != MAGIC or not 0 < metadata_size <= 4096 or not 0 < stream_size <= MAX_STREAM:
        raise ValueError('Invalid FOW DCVC frame header')
    meta = json.loads(read_exact(sock, metadata_size))
    validate_metadata(meta)
    return meta, read_exact(sock, stream_size)


class DCVCCodec:
    def __init__(self, args):
        args = resolve_codec_paths(args)
        import torch
        self.torch = torch
        if not args.device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('DCVC-RT requires an NVIDIA CUDA device; CPU/MPS are not supported by this experiment.')
        root = args.dcvc_root.expanduser().resolve()
        if not (root / 'src/models/video_model.py').is_file():
            raise ValueError(f'DCVC-RT not found at {root}. Follow the README setup, '
                             'edit DCVC_ROOT in dcvc_common.py, or pass --dcvc-root.')
        for path in (args.model_i, args.model_p):
            if not path.expanduser().is_file():
                raise FileNotFoundError(f'Missing checkpoint: {path}')
        sys.path.insert(0, str(root))
        try:
            from src.models.image_model import DMCI
            from src.models.video_model import DMC
            from src.utils.common import get_state_dict, set_torch_env
            from src.utils.transforms import rgb2ycbcr, ycbcr2rgb
            from src.layers.cuda_inference import replicate_pad
            set_torch_env()
            self.device = torch.device(args.device)
            torch.cuda.set_device(self.device)
            self.intra = DMCI()
            self.inter = DMC()
            for model, checkpoint in ((self.intra, args.model_i), (self.inter, args.model_p)):
                model.load_state_dict(get_state_dict(str(checkpoint.expanduser())))
                model.to(self.device).eval()
                model.update(None)
                model.half()
        except ImportError as exc:
            raise RuntimeError('Build the DCVC-RT entropy coder and CUDA extensions; see the README setup.') from exc
        self.to_yuv = rgb2ycbcr
        self.to_rgb = ycbcr2rgb
        self.pad = replicate_pad
        self.reset()

    def reset(self):
        self.inter.clear_dpb()
        self.inter.set_curr_poc(0)
        self.next_frame = 0
        self.dimensions = None

    def configure(self, width, height):
        dimensions = (width, height)
        if self.dimensions is not None and self.dimensions != dimensions:
            raise ValueError('Resolution changed during the stream; reconnect to reset the codec')
        self.dimensions = dimensions
        two_coders = width * height > 1280 * 720
        self.intra.set_use_two_entropy_coders(two_coders)
        self.inter.set_use_two_entropy_coders(two_coders)
        return {'sps_id': 0, 'width': width, 'height': height,
                'ec_part': int(two_coders), 'use_ada_i': 0}

    def encode(self, rgb, qp, intra_period):
        torch = self.torch
        height, width = rgb.shape[:2]
        self.configure(width, height)
        index = self.next_frame
        is_intra = index % intra_period == 0
        # Follow upstream's temporal quantizer schedule; periodic I frames reset
        # references, so this experiment disables adaptive feature refresh.
        qp = qp if is_intra else self.inter.shift_qp(qp, [0, 1, 0, 2, 0, 2, 0, 2][index % 8])
        with torch.inference_mode():
            x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0)
            x = self.to_yuv(x.to(self.device, dtype=torch.float32) / 255).half()
            right, bottom = self.intra.get_padding_size(height, width, 16)
            x = self.pad(x, bottom, right)
            encoded = (self.intra if is_intra else self.inter).compress(x, qp)
            if is_intra:
                self.inter.clear_dpb()
                self.inter.add_ref_frame(None, encoded['x_hat'])
        self.next_frame += 1
        return {'frame': index, 'kind': 'I' if is_intra else 'P',
                'width': width, 'height': height, 'qp': qp}, encoded['bit_stream']

    def decode(self, meta, stream):
        validate_metadata(meta)
        if meta['frame'] != self.next_frame or (self.next_frame == 0 and meta['kind'] != 'I'):
            raise ValueError('Missing/out-of-order reference frame; reconnect the sender')
        sps = self.configure(meta['width'], meta['height'])
        with self.torch.inference_mode():
            model = self.intra if meta['kind'] == 'I' else self.inter
            decoded = model.decompress(stream, sps, meta['qp'])
            if meta['kind'] == 'I':
                self.inter.clear_dpb()
                self.inter.add_ref_frame(None, decoded['x_hat'])
            x = decoded['x_hat'][:, :, :meta['height'], :meta['width']]
            rgb = self.to_rgb(x).clamp(0, 1).mul(255).round().to(self.torch.uint8)
            frame = rgb[0].permute(1, 2, 0).cpu().numpy()
        self.next_frame += 1
        return frame

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)
