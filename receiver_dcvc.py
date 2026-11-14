"""FOW TCP receiver for DCVC-RT compressed camera/video frames."""
import argparse
import queue
import socket
import threading
import time

from dcvc_common import DCVCCodec, add_codec_args, receive_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_codec_args(parser)
    parser.add_argument('--bind', default='0.0.0.0')
    args = parser.parse_args()
    import cv2
    import numpy as np
    codec = DCVCCodec(args)
    outputs = queue.Queue(maxsize=1)
    stop = threading.Event()
    # Decode EVERY frame to preserve references, but only display the latest.
    def receive(conn):
        try:
            while not stop.is_set():
                item = receive_frame(conn)
                if item is None:
                    break
                meta, stream = item
                codec.synchronize()
                started = time.perf_counter()
                rgb = codec.decode(meta, stream)
                codec.synchronize()
                label = f"{meta['kind']} | {len(stream)/1024:.1f} KiB payload | {(time.perf_counter()-started)*1000:.0f} ms"
                print(f"[Receiver] Frame {meta['frame']}: {label}")
                try:
                    outputs.get_nowait()
                except queue.Empty:
                    pass
                outputs.put_nowait((rgb, label))
        except Exception as exc:
            print(f'[Receiver Error] {exc}')
        finally:
            stop.set()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.bind, args.port))
        server.listen(1)
        server.settimeout(0.1)
        print(f'[Receiver] Listening on TCP {args.bind}:{args.port}; Ctrl+C to stop')
        conn = None
        worker = None
        panel = np.full((240, 640, 3), 40, dtype=np.uint8)
        cv2.putText(panel, 'Waiting for DCVC sender...', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        try:
            while True:
                if conn is None:
                    try:
                        conn, addr = server.accept()
                        print(f'[Receiver] Connected: {addr}')
                        worker = threading.Thread(target=receive, args=(conn,), daemon=True)
                        worker.start()
                    except socket.timeout:
                        pass
                try:
                    rgb, label = outputs.get_nowait()
                    panel = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.putText(panel, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                except queue.Empty:
                    pass
                if not args.headless:
                    cv2.imshow('DCVC reconstructed', panel)
                    if cv2.waitKey(1) & 0xff in (ord('q'), 27):
                        break
                else:
                    time.sleep(0.01)
                if stop.is_set():
                    break
        finally:
            stop.set()
            if conn is not None:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                conn.close()
            if worker is not None:
                worker.join(timeout=1)
            if not args.headless:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
