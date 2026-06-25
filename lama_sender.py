import cv2
import numpy as np
import socket
import pickle
import random
import math
import time

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

FRAME_W = 320
FRAME_H = 240

PATCH_SIZE = 16

VISIBLE_PATCH_RATIO = 0.10
KEYFRAME_PERIOD = 30

MAX_PATCHES_PER_PACKET = 4
KEYFRAME_CHUNK_SIZE = 1000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(1)

frame_id = 0

all_positions = [
    (x, y)
    for y in range(0, FRAME_H, PATCH_SIZE)
    for x in range(0, FRAME_W, PATCH_SIZE)
]

num_selected = max(
    1,
    int(len(all_positions) * VISIBLE_PATCH_RATIO)
)

print(
    f"total patches={len(all_positions)} "
    f"selected={num_selected}"
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(
        frame,
        (FRAME_W, FRAME_H)
    )

    #
    # KEYFRAME
    #
    if frame_id % KEYFRAME_PERIOD == 0:

        success, jpg = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                60
            ]
        )

        if not success:
            continue

        jpg_bytes = jpg.tobytes()

        total_chunks = math.ceil(
            len(jpg_bytes)
            / KEYFRAME_CHUNK_SIZE
        )

        print(
            f"KEYFRAME {frame_id}: "
            f"{len(jpg_bytes)} bytes "
            f"{total_chunks} chunks"
        )

        for chunk_idx in range(total_chunks):

            start = (
                chunk_idx
                * KEYFRAME_CHUNK_SIZE
            )

            end = start + KEYFRAME_CHUNK_SIZE

            payload = {
                "type": "keyframe_chunk",
                "frame_id": frame_id,
                "chunk_idx": chunk_idx,
                "total_chunks": total_chunks,
                "data": jpg_bytes[start:end]
            }

            sock.sendto(
                pickle.dumps(payload),
                (UDP_IP, UDP_PORT)
            )

    #
    # DELTA FRAME
    #
    else:

        selected_positions = random.sample(
            all_positions,
            num_selected
        )

        patches = []

        for x, y in selected_positions:

            patch = frame[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            patches.append(
                (
                    x,
                    y,
                    patch.tobytes()
                )
            )

        total_chunks = math.ceil(
            len(patches)
            / MAX_PATCHES_PER_PACKET
        )

        print(
            f"DELTA {frame_id}: "
            f"{len(patches)} patches "
            f"{total_chunks} chunks"
        )

        for chunk_idx in range(total_chunks):

            start = (
                chunk_idx
                * MAX_PATCHES_PER_PACKET
            )

            end = start + MAX_PATCHES_PER_PACKET

            payload = {
                "type": "delta",
                "frame_id": frame_id,
                "chunk_idx": chunk_idx,
                "total_chunks": total_chunks,
                "patches": patches[start:end]
            }

            packet = pickle.dumps(payload)

            sock.sendto(
                packet,
                (UDP_IP, UDP_PORT)
            )

    cv2.imshow(
        "Sender",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

    frame_id += 1

cap.release()
cv2.destroyAllWindows()