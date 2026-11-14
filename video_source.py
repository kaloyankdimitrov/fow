"""Shared camera/file selection for the UDP senders."""
import argparse

import cv2


def parse_source_args():
    parser = argparse.ArgumentParser(description="Stream webcam or recorded video frames.")
    parser.add_argument("--input", metavar="PATH", help="Video file instead of a webcam")
    parser.add_argument("--camera", type=int, default=None, metavar="INDEX",
                        help="Camera index (default: 1; some senders fall back to 0)")
    return parser.parse_args()


def open_source(args, fallback_camera=False):
    source = args.input if args.input is not None else (
        args.camera if args.camera is not None else 1
    )
    cap = cv2.VideoCapture(source)
    if not cap.isOpened() and args.input is None and args.camera is None and fallback_camera:
        cap.release()
        print("Camera index 1 not found. Falling back to camera index 0...")
        source = 0
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video source: {source!r}")
    return cap
