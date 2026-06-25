import cv2
import numpy as np
import socket
import pickle
import time

from simple_lama_inpainting import SimpleLama

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

FRAME_W = 320
FRAME_H = 240
PATCH_SIZE = 16

FRAME_TIMEOUT = 0.5

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

lama = SimpleLama()

reference = None

pending_keyframes = {}
pending_deltas = {}

while True:

    try:
        data, _ = sock.recvfrom(65535)
        packet = pickle.loads(data)

    except Exception as e:
        print("recv error:", e)
        continue

    now = time.time()

    #
    # cleanup stale frames
    #
    for d in (pending_keyframes, pending_deltas):

        expired = []

        for fid, info in d.items():
            if now - info["timestamp"] > FRAME_TIMEOUT:
                expired.append(fid)

        for fid in expired:
            del d[fid]

    packet_type = packet.get("type")

    #
    # KEYFRAME CHUNK
    #
    if packet_type == "keyframe_chunk":

        fid = packet["frame_id"]

        if fid not in pending_keyframes:

            pending_keyframes[fid] = {
                "timestamp": now,
                "total_chunks": packet["total_chunks"],
                "chunks": {}
            }

        pending_keyframes[fid]["chunks"][
            packet["chunk_idx"]
        ] = packet["data"]

        info = pending_keyframes[fid]

        if len(info["chunks"]) != info["total_chunks"]:
            continue

        try:

            jpg_bytes = b"".join(
                info["chunks"][i]
                for i in range(info["total_chunks"])
            )

            jpg_np = np.frombuffer(
                jpg_bytes,
                dtype=np.uint8
            )

            frame = cv2.imdecode(
                jpg_np,
                cv2.IMREAD_COLOR
            )

            if frame is None:
                print("failed to decode keyframe")
                del pending_keyframes[fid]
                continue

            reference = frame

            print(
                f"decoded keyframe {fid}"
            )

            cv2.imshow(
                "Reconstructed",
                reference
            )

            cv2.waitKey(1)

        except Exception as e:

            print(
                "keyframe decode error:",
                e
            )

        del pending_keyframes[fid]

        continue

    #
    # DELTA
    #
    if packet_type == "delta":

        if reference is None:
            continue

        fid = packet["frame_id"]

        if fid not in pending_deltas:

            pending_deltas[fid] = {
                "timestamp": now,
                "total_chunks":
                    packet["total_chunks"],
                "received": {}
            }

        pending_deltas[fid]["received"][
            packet["chunk_idx"]
        ] = packet["patches"]

        frame_info = pending_deltas[fid]

        if len(frame_info["received"]) != frame_info["total_chunks"]:
            continue

        sparse = reference.copy()

        missing_mask = np.ones(
            (FRAME_H, FRAME_W),
            dtype=np.uint8
        ) * 255

        try:

            for chunk_idx in range(
                frame_info["total_chunks"]
            ):

                patches = frame_info[
                    "received"
                ][chunk_idx]

                for x, y, patch_bytes in patches:

                    expected_size = (
                        PATCH_SIZE
                        * PATCH_SIZE
                        * 3
                    )

                    if len(patch_bytes) != expected_size:
                        continue

                    patch = np.frombuffer(
                        patch_bytes,
                        dtype=np.uint8
                    ).reshape(
                        PATCH_SIZE,
                        PATCH_SIZE,
                        3
                    )

                    sparse[
                        y:y+PATCH_SIZE,
                        x:x+PATCH_SIZE
                    ] = patch

                    missing_mask[
                        y:y+PATCH_SIZE,
                        x:x+PATCH_SIZE
                    ] = 0

            sparse_rgb = cv2.cvtColor(
                sparse,
                cv2.COLOR_BGR2RGB
            )

            result_rgb = np.array(
                lama(
                    sparse_rgb,
                    missing_mask
                )
            )

            result_bgr = cv2.cvtColor(
                result_rgb,
                cv2.COLOR_RGB2BGR
            )

            reference = result_bgr

            cv2.imshow(
                "Sparse",
                sparse
            )

            cv2.imshow(
                "Mask",
                missing_mask
            )

            cv2.imshow(
                "Reconstructed",
                result_bgr
            )

            cv2.waitKey(1)

        except Exception as e:

            print(
                "delta reconstruction error:",
                e
            )

        del pending_deltas[fid]