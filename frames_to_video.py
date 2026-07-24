#!/usr/bin/env python3
"""
Convert camera frame images to video with frame numbers displayed.
Usage: python frames_to_video.py --input-dir <camera_frames_dir> --output <output.mp4> [--fps 30]
"""

import argparse
import cv2
import os
from pathlib import Path
from natsort import natsorted


def frames_to_video(input_dir, output_file, fps=30, font_scale=1.5, font_color=(0, 255, 0)):
    """
    Convert image frames to video with frame numbers.

    Args:
        input_dir: Directory containing image frames
        output_file: Output video file path
        fps: Frames per second for output video
        font_scale: Font size scale for frame numbers
        font_color: BGR color tuple for frame number text
    """
    input_path = Path(input_dir)

    # Get all image files sorted naturally
    image_files = natsorted([
        f for f in input_path.glob('*')
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']
    ])

    if not image_files:
        print(f"No image files found in {input_dir}")
        return False

    print(f"Found {len(image_files)} images")

    # Read first frame to get dimensions
    first_frame = cv2.imread(str(image_files[0]))
    if first_frame is None:
        print(f"Failed to read first frame: {image_files[0]}")
        return False

    height, width = first_frame.shape[:2]
    print(f"Frame dimensions: {width}x{height}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Failed to initialize video writer for {output_file}")
        return False

    # Process each frame
    for frame_num, image_path in enumerate(image_files):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Warning: Failed to read {image_path}, skipping")
            continue

        # Add frame number to top-left corner
        text = f"Frame {frame_num}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2

        # Get text size for background rectangle
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x, text_y = 10, 30

        # Draw semi-transparent background rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay,
                     (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Put text
        cv2.putText(frame, text, (text_x, text_y), font, font_scale,
                   font_color, thickness, cv2.LINE_AA)

        # Write frame to video
        out.write(frame)

        if (frame_num + 1) % 100 == 0:
            print(f"Processed {frame_num + 1}/{len(image_files)} frames")

    out.release()
    print(f"✓ Video saved to {output_file}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert image frames to video with frame numbers')
    parser.add_argument('--input-dir', '-i', required=True, help='Directory containing image frames')
    parser.add_argument('--output', '-o', required=True, help='Output video file path')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second (default: 30)')
    parser.add_argument('--font-scale', type=float, default=1.5, help='Font size scale (default: 1.5)')

    args = parser.parse_args()

    success = frames_to_video(args.input_dir, args.output, fps=args.fps, font_scale=args.font_scale)
    exit(0 if success else 1)
