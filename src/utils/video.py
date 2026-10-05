from pathlib import Path

import numpy as np
from PIL import Image


def frames_to_images(frames: list) -> list:
    """
    Convert RGB frames produced by env.render() (render_mode="rgb_array") to PIL images

    Args:
        frames (list): list of np.array frames of shape (H, W, 3)

    Returns:
        list: list of PIL.Image in RGB mode
    """
    if len(frames) == 0:
        raise ValueError("No frames to convert")

    images = []
    for i, frame in enumerate(frames):
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError(f"Frame {i} has shape {frame.shape}, expected (H, W, 3) or (H, W, 4)")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(np.ascontiguousarray(frame[:, :, :3])))

    return images


def save_gif(frames: list, path, fps=4, loop=0, hold_last=2):
    """
    Create and save a .gif file from a list of frames

    Args:
        frames (list): list of np.array RGB frames from env.render()
        path (str | Path): output file path, ".gif" is appended if missing
        fps (int): frames per second, env.metadata["render_fps"] is a good default
        loop (int): number of times the gif loops, 0 loops forever
        hold_last (int): extra number of frames to hold the final frame for, so the end state is visible

    Returns:
        Path: path the gif was saved to
    """
    path = Path(path)
    if path.suffix.lower() != ".gif":
        path = path.with_suffix(".gif")
    path.parent.mkdir(parents=True, exist_ok=True)

    images = frames_to_images(frames)

    # Duration of each frame in milliseconds, last frame is held longer
    frame_duration = int(1000 / fps)
    durations = [frame_duration] * len(images)
    durations[-1] = frame_duration * (1 + hold_last)

    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=loop,
        optimize=False,
    )

    return path
