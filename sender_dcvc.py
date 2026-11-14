"""Camera/file -> DCVC-RT bitstream -> FOW TCP receiver."""
import argparse
import socket
import time

from dcvc_common import DCVCCodec, add_codec_args, positive, send_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_codec_args(parser)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--input', help='Video file instead of a camera')
    parser.add_argument('--width', type=positive, default=320)
    parser.add_argument('--height', type=positive, default=240)
    parser.add_argument('--fps', type=positive, default=30)
    parser.add_argument('--qp', type=int, choices=range(64), default=32, metavar='0..63')
    parser.add_argument('--intra-period', type=positive, default=32)
    parser.add_argument('--frames', type=positive, help='Stop after this many frames')
    args = parser.parse_args()
    if max(args.width, args.height) > 4096:
        parser.error('Maximum supported width/height is 4096')
    import cv2
    codec = DCVCCodec(args)
    cap = cv2.VideoCapture(args.input if args.input else args.camera)
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open camera/video source')
        with socket.create_connection((args.host, args.port), timeout=30) as sock:
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f'[Sender] DCVC-RT over TCP to {args.host}:{args.port}; Ctrl+C to stop')
            while args.frames is None or codec.next_frame < args.frames:
                started = time.perf_counter()
                ok, bgr = cap.read()
                if not ok:
                    break
                bgr = cv2.resize(bgr, (args.width, args.height))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                codec.synchronize()
                encode_start = time.perf_counter()
                meta, stream = codec.encode(rgb, args.qp, args.intra_period)
                codec.synchronize()
                elapsed = time.perf_counter() - encode_start
                size = send_frame(sock, meta, stream)
                label = f"{meta['kind']} QP {meta['qp']} | {size / 1024:.1f} KiB | {elapsed * 1000:.0f} ms"
                print(f"[Sender] Frame {meta['frame']}: {label}")
                if not args.headless:
                    cv2.putText(bgr, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    cv2.imshow('DCVC source', bgr)
                    if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                        break
                time.sleep(max(0, 1 / args.fps - (time.perf_counter() - started)))
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
