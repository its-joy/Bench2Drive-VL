"""
make_event_contact_sheet.py

Create labeled contact sheets for GT post-action events.

Example:
    python3 make_event_contact_sheet.py \
      --route-dir eval_v1/Qwen2.5VL+front_cam/RouteScenario_0_rep0_Town10HD_SignalizedJunctionRightTurn_Weather0_07_08_14_38_08 \
      --gt-log gt_logs/RouteScenario_0_rep0_Town10HD_SignalizedJunctionRightTurn_Weather0_07_08_14_38_08.json \
      --event-index 5 \
      --camera-name rgb_front

By default, the script writes one sheet per event.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).parent
FRAME_RATE = 10


def load_font(size: int, family: str = "arial", bold: bool = False) -> ImageFont.ImageFont:
    family_candidates = {
        "arial": [
            "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf" if bold else "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf" if bold else "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ],
        "calibri": [
            "/usr/share/fonts/truetype/msttcorefonts/Calibri_Bold.ttf" if bold else "/usr/share/fonts/truetype/msttcorefonts/Calibri.ttf",
            "/usr/share/fonts/truetype/msttcorefonts/calibrib.ttf" if bold else "/usr/share/fonts/truetype/msttcorefonts/calibri.ttf",
            "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf" if bold else "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ],
    }
    candidates = family_candidates.get(family, family_candidates["arial"])
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    if not words:
        return []

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def sanitize_filename(text: str) -> str:
    keep = []
    for ch in text.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "_", "-", "+"}:
            keep.append("_")
    return "_".join(part for part in "".join(keep).split("_") if part).strip("_") or "event"


def load_events(gt_log: Path) -> List[Dict]:
    data = json.load(open(gt_log))
    events = data.get("episode", {}).get("events", [])
    if not events:
        raise RuntimeError(f"No episode events found in {gt_log}")
    return events


def available_frames(images_dir: Path) -> Dict[int, Path]:
    frames = {}
    for path in images_dir.glob("*.jpg"):
        try:
            frames[int(path.stem)] = path
        except ValueError:
            continue
    if not frames:
        raise RuntimeError(f"No .jpg frames found in {images_dir}")
    return frames


def sample_event_frames(
    frames: Dict[int, Path],
    frame_start: int,
    frame_end: int,
    stride: int,
    max_frames: Optional[int],
    include_end: bool = True,
) -> List[Tuple[int, Path]]:
    in_range = [fn for fn in sorted(frames) if frame_start <= fn <= frame_end]
    if not in_range:
        return []

    sampled = in_range[::max(1, stride)]
    if include_end and frame_end in frames and frame_end not in sampled:
        sampled.append(frame_end)
        sampled = sorted(set(sampled))

    if max_frames and len(sampled) > max_frames:
        indices = [
            round(i * (len(sampled) - 1) / (max_frames - 1))
            for i in range(max_frames)
        ]
        sampled = [sampled[i] for i in sorted(set(indices))]

    return [(fn, frames[fn]) for fn in sampled]


def resize_to_tile(image: Image.Image, tile_w: int, tile_h: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(tile_w / image.width, tile_h / image.height)
    new_w = max(1, int(image.width * scale))
    new_h = max(1, int(image.height * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_w, tile_h), (18, 18, 18))
    tile.paste(resized, ((tile_w - new_w) // 2, (tile_h - new_h) // 2))
    return tile


def route_dir_from_frame_path(frame_path: Path) -> Optional[Path]:
    """Infer route dir from .../<route>/camera/<camera_name>/<frame>.jpg."""
    for parent in frame_path.parents:
        if parent.name == "camera":
            return parent.parent
    return None


def load_frame_speed_raw(frame_number: int, frame_path: Path) -> Optional[float]:
    route_dir = route_dir_from_frame_path(frame_path)
    if route_dir is None:
        return None

    anno_path = route_dir / "anno" / f"{frame_number:05d}.json"
    if not anno_path.exists():
        return None

    try:
        anno = json.load(open(anno_path))
    except (OSError, json.JSONDecodeError):
        return None

    speed = anno.get("speed")
    if speed is None:
        for box in anno.get("bounding_boxes", []):
            if isinstance(box, dict) and box.get("class") == "ego_vehicle":
                speed = box.get("speed")
                break
    try:
        return float(speed) if speed is not None else None
    except (TypeError, ValueError):
        return None


def interpolated_event_speed_kmh(frame_number: int, event: Dict) -> Optional[float]:
    speed_start = event.get("speed_start_kmh")
    speed_end = event.get("speed_end_kmh")
    frame_start = event.get("frame_start")
    frame_end = event.get("frame_end", frame_start)
    if speed_start is None or speed_end is None or frame_start is None or frame_end is None:
        return None
    try:
        frame_start = int(frame_start)
        frame_end = int(frame_end)
        speed_start = float(speed_start)
        speed_end = float(speed_end)
    except (TypeError, ValueError):
        return None
    if frame_end <= frame_start:
        return speed_start
    ratio = (frame_number - frame_start) / (frame_end - frame_start)
    ratio = max(0.0, min(1.0, ratio))
    return speed_start + (speed_end - speed_start) * ratio


def normalize_speed_to_kmh(raw_speed: float, expected_kmh: Optional[float]) -> float:
    """Annotation speed may be m/s; choose km/h scale using event speed as reference."""
    if expected_kmh is None:
        return raw_speed * 3.6 if abs(raw_speed) < 20.0 else raw_speed
    raw_as_kmh = raw_speed
    converted_as_kmh = raw_speed * 3.6
    if abs(converted_as_kmh - expected_kmh) < abs(raw_as_kmh - expected_kmh):
        return converted_as_kmh
    return raw_as_kmh


def frame_speed_label(frame_number: int, frame_path: Path, event: Dict) -> Optional[str]:
    expected_speed = interpolated_event_speed_kmh(frame_number, event)
    raw_speed = load_frame_speed_raw(frame_number, frame_path)
    if raw_speed is not None:
        speed_kmh = normalize_speed_to_kmh(raw_speed, expected_speed)
    else:
        speed_kmh = expected_speed
    if speed_kmh is None:
        return None
    return f"{speed_kmh:.1f} km/h"


def draw_label(
    tile: Image.Image,
    label: str,
    font: ImageFont.ImageFont,
    bg: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    draw = ImageDraw.Draw(tile, "RGBA")
    font_size = getattr(font, "size", 18)
    pad_x = max(8, int(font_size * 0.45))
    pad_y = max(5, int(font_size * 0.28))
    line_gap = max(3, int(font_size * 0.12))
    lines = label.splitlines() or [label]
    line_sizes = [text_size(draw, line, font) for line in lines]
    text_w = max(width for width, _ in line_sizes)
    text_h = sum(height for _, height in line_sizes) + max(0, len(lines) - 1) * line_gap
    draw.rectangle(
        (0, 0, text_w + pad_x * 2, text_h + pad_y * 2),
        fill=(*bg, 220),
    )
    y = pad_y
    for line, (_, line_h) in zip(lines, line_sizes):
        draw.text((pad_x, y), line, fill=(255, 255, 255, 255), font=font)
        y += line_h + line_gap


def event_title(
    event_index: int,
    event: Dict,
    frame_count: int,
    camera_name: str,
    context: str = "high",
) -> str:
    frame_start = event.get("frame_start")
    frame_end = event.get("frame_end", frame_start)
    t_s = event.get("t_s", frame_start / FRAME_RATE if frame_start is not None else 0)
    t_end_s = event.get("t_end_s", frame_end / FRAME_RATE if frame_end is not None else t_s)
    event_label = f"Event {event_index}"
    if context == "high":
        action = event.get("ego_action", [])
        action_text = " + ".join(action) if isinstance(action, list) else str(action)
        event_label = f"{event_label}: {action_text}"
    return (
        f"{event_label} | frames {frame_start}-{frame_end} "
        f"| t={t_s:.1f}-{t_end_s:.1f}s | {frame_count} sampled | {camera_name}"
    )


def make_contact_sheet(
    sampled: List[Tuple[int, Path]],
    event_index: int,
    event: Dict,
    camera_name: str,
    out_path: Path,
    cols: int,
    tile_width: int,
    gap: int,
    header_height: int,
    title_font_size: int = 32,
    metadata_font_size: int = 22,
    frame_label_font_size: int = 28,
    font_family: str = "arial",
    context: str = "high",
) -> None:
    if not sampled:
        raise RuntimeError("Cannot make a contact sheet with no sampled frames")

    first_img = Image.open(sampled[0][1])
    aspect = first_img.height / first_img.width
    tile_w = tile_width
    tile_h = max(1, int(tile_w * aspect))
    rows = math.ceil(len(sampled) / cols)
    sheet_w = cols * tile_w + (cols + 1) * gap

    title_font = load_font(title_font_size, font_family, bold=True)
    label_font = load_font(frame_label_font_size, font_family, bold=True)
    small_font = load_font(metadata_font_size, font_family)

    title = event_title(event_index, event, len(sampled), camera_name, context=context)
    speed_start = event.get("speed_start_kmh")
    speed_end = event.get("speed_end_kmh")
    lane_id = event.get("lane_id")
    traffic_light = event.get("traffic_light_state")
    details = []
    if context == "high" and speed_start is not None and speed_end is not None:
        details.append(f"speed {speed_start:.1f}-{speed_end:.1f} km/h")
    if context == "high" and lane_id is not None:
        details.append(f"lane {lane_id}")
    if context == "high" and traffic_light:
        details.append(f"traffic light {traffic_light}")
    metadata = " | ".join(details)

    measure = Image.new("RGB", (1, 1))
    measure_draw = ImageDraw.Draw(measure)
    max_text_width = max(1, sheet_w - gap * 2)
    title_lines = wrap_text(measure_draw, title, title_font, max_text_width)
    metadata_lines = wrap_text(measure_draw, metadata, small_font, max_text_width) if metadata else []
    title_line_heights = [text_size(measure_draw, line, title_font)[1] for line in title_lines]
    metadata_line_heights = [text_size(measure_draw, line, small_font)[1] for line in metadata_lines]
    title_h = sum(title_line_heights) + max(0, len(title_lines) - 1) * max(4, int(title_font_size * 0.06))
    metadata_h = sum(metadata_line_heights) + max(0, len(metadata_lines) - 1) * max(4, int(metadata_font_size * 0.06))
    top_pad = max(8, int(title_font_size * 0.08))
    line_gap = max(6, int(metadata_font_size * 0.12)) if metadata else 0
    bottom_pad = max(10, int(metadata_font_size * 0.18))
    compact_header_height = top_pad + title_h + line_gap + metadata_h + bottom_pad
    actual_header_height = max(compact_header_height, gap)

    sheet_h = actual_header_height + rows * tile_h + (rows + 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    y_cursor = top_pad
    title_line_gap = max(4, int(title_font_size * 0.06))
    for line, line_h in zip(title_lines, title_line_heights):
        draw.text((gap, y_cursor), line, fill=(20, 20, 20), font=title_font)
        y_cursor += line_h + title_line_gap
    y_cursor = y_cursor - title_line_gap + line_gap
    metadata_line_gap = max(4, int(metadata_font_size * 0.06))
    for line, line_h in zip(metadata_lines, metadata_line_heights):
        draw.text((gap, y_cursor), line, fill=(45, 45, 45), font=small_font)
        y_cursor += line_h + metadata_line_gap

    for idx, (frame_number, img_path) in enumerate(sampled):
        row = idx // cols
        col = idx % cols
        x = gap + col * (tile_w + gap)
        y = actual_header_height + gap + row * (tile_h + gap)
        with Image.open(img_path) as img:
            tile = resize_to_tile(img, tile_w, tile_h)
        speed_label = frame_speed_label(frame_number, img_path, event)
        label = f"frame {frame_number}"
        if speed_label:
            label += f"\n{speed_label}"
        draw_label(tile, label, label_font)
        sheet.paste(tile, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=100, subsampling=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create contact sheets for GT event frame windows."
    )
    parser.add_argument("--route-dir", required=True, help="Route directory containing camera frames")
    parser.add_argument("--gt-log", required=True, help="GT event log JSON")
    parser.add_argument("--event-index", type=int, default=None,
                        help="1-based event index to render. Omit to render all events.")
    parser.add_argument("--camera-name", default="rgb_front",
                        help="Camera folder under route_dir/camera (default: rgb_front)")
    parser.add_argument("--out-dir", default="output/contact_sheets",
                        help="Output directory")
    parser.add_argument("--frame-stride", type=int, default=5,
                        help="Sample every Nth frame before max-frame reduction")
    parser.add_argument("--max-frames", type=int, default=20,
                        help="Maximum tiles per sheet")
    parser.add_argument("--cols", type=int, default=5,
                        help="Number of columns in the sheet")
    parser.add_argument("--tile-width", type=int, default=640,
                        help="Width of each tile in pixels")
    parser.add_argument("--gap", type=int, default=8,
                        help="Gap between tiles in pixels")
    parser.add_argument("--header-height", type=int, default=260,
                        help="Header height in pixels")
    parser.add_argument("--title-font-size", type=int, default=160,
                        help="Font size for the contact-sheet title")
    parser.add_argument("--metadata-font-size", type=int, default=120,
                        help="Font size for the contact-sheet metadata line")
    parser.add_argument("--frame-label-font-size", type=int, default=180,
                        help="Font size for each frame-number label")
    parser.add_argument("--font-family", default="arial",
                        choices=["arial", "calibri"],
                        help="Font family style for labels. Arial uses Liberation Sans fallback when Arial is unavailable.")
    parser.add_argument("--context", default="high",
                        choices=["high", "low"],
                        help="high includes action/lane/traffic-light metadata; low keeps only neutral frame metadata")
    parser.add_argument("--no-include-end", action="store_true",
                        help="Do not force the event end frame into the sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    route_dir = Path(args.route_dir)
    gt_log = Path(args.gt_log)
    camera_dir = route_dir / "camera" / args.camera_name
    out_root = Path(args.out_dir) / route_dir.name

    events = load_events(gt_log)
    frames = available_frames(camera_dir)

    if args.event_index is not None:
        if args.event_index < 1 or args.event_index > len(events):
            raise SystemExit(f"--event-index must be between 1 and {len(events)}")
        selected = [(args.event_index, events[args.event_index - 1])]
    else:
        selected = list(enumerate(events, start=1))

    for event_index, event in selected:
        frame_start = int(event.get("frame_start", 0))
        frame_end = int(event.get("frame_end", frame_start))
        sampled = sample_event_frames(
            frames,
            frame_start=frame_start,
            frame_end=frame_end,
            stride=args.frame_stride,
            max_frames=args.max_frames,
            include_end=not args.no_include_end,
        )
        if not sampled:
            print(f"[SKIP] Event {event_index}: no frames in {frame_start}-{frame_end}")
            continue

        if args.context == "high":
            action = event.get("ego_action", [])
            action_text = " + ".join(action) if isinstance(action, list) else str(action)
            out_name = (
                f"event_{event_index:02d}_{sanitize_filename(action_text)}_"
                f"frames_{frame_start:05d}_{frame_end:05d}_{args.camera_name}.jpg"
            )
        else:
            out_name = (
                f"event_{event_index:02d}_"
                f"frames_{frame_start:05d}_{frame_end:05d}_{args.camera_name}.jpg"
            )
        out_path = out_root / out_name
        make_contact_sheet(
            sampled=sampled,
            event_index=event_index,
            event=event,
            camera_name=args.camera_name,
            out_path=out_path,
            cols=args.cols,
            tile_width=args.tile_width,
            gap=args.gap,
            header_height=args.header_height,
            title_font_size=args.title_font_size,
            metadata_font_size=args.metadata_font_size,
            frame_label_font_size=args.frame_label_font_size,
            font_family=args.font_family,
            context=args.context,
        )
        first_frame = sampled[0][0]
        last_frame = sampled[-1][0]
        print(
            f"[OK] Event {event_index}: {len(sampled)} frames "
            f"({first_frame}-{last_frame}) -> {out_path}"
        )


if __name__ == "__main__":
    main()
