"""
generate_video.py
~~~~~~~~~~~~~~~~~
Generate a lyrics video from synced LRC files using Pillow + ffmpeg.

Shows Vietnamese lyrics (bold, white) with English translation (italic, yellow)
centered on a dark background, with the original audio track.

Dependencies: Pillow, ffmpeg (basic install — no libass needed)
"""

import json
import math
import re
import subprocess
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# LRC parsing
# ---------------------------------------------------------------------------
_LRC_RE = re.compile(r"^\[(\d{2}):(\d{2})\.(\d{2,3})\]\s*(.*)")


def parse_lrc(lrc_text: str) -> list[dict]:
    """Parse LRC into [{start_s, text}, …] sorted by time."""
    events = []
    for line in lrc_text.strip().splitlines():
        m = _LRC_RE.match(line)
        if not m:
            continue
        mins, secs, frac = int(m.group(1)), int(m.group(2)), m.group(3)
        ms = int(frac) * 10 if len(frac) == 2 else int(frac)
        start = mins * 60 + secs + ms / 1000.0
        events.append({"start_s": start, "text": m.group(4).strip()})
    events.sort(key=lambda e: e["start_s"])
    return events


def _compute_end_times(events: list[dict]) -> list[float]:
    """End time = next event's start, or +5 s for last."""
    ends = []
    for i, ev in enumerate(events):
        ends.append(events[i + 1]["start_s"] if i + 1 < len(events) else ev["start_s"] + 5.0)
    return ends


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------
def _load_font(font_path: str | None, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a TTF/OTF font, falling back to Pillow's default."""
    if font_path:
        p = Path(font_path)
        if p.is_dir():
            if bold:
                candidates = ["NotoSans-Bold.ttf", "NotoSans-SemiBold.ttf"]
            else:
                candidates = ["NotoSans-Regular.ttf", "NotoSans-Medium.ttf"]
            for c in candidates:
                fp = p / c
                if fp.exists():
                    return ImageFont.truetype(str(fp), size)
            ttfs = list(p.glob("*.ttf"))
            if ttfs:
                return ImageFont.truetype(str(ttfs[0]), size)
        elif p.exists():
            return ImageFont.truetype(str(p), size)
    try:
        return ImageFont.truetype("Arial", size)
    except OSError:
        return ImageFont.load_default(size)


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------
_BG_COLOR = (26, 26, 46)        # dark navy
_VI_COLOR = (255, 255, 255)      # white
_EN_COLOR = (255, 255, 100)      # soft yellow
_TITLE_COLOR = (160, 160, 190)   # muted lavender


def _draw_text_centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    y: int,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    width: int,
):
    """Draw text horizontally centered at vertical position *y* with shadow."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (width - tw) // 2
    # Shadow for readability
    for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2), (0, 2)]:
        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)


def render_frame(
    width: int,
    height: int,
    vi_text: str,
    en_text: str,
    title: str,
    vi_font: ImageFont.FreeTypeFont,
    en_font: ImageFont.FreeTypeFont,
    title_font: ImageFont.FreeTypeFont,
) -> bytes:
    """Render a single video frame as raw RGB bytes."""
    img = Image.new("RGB", (width, height), _BG_COLOR)
    draw = ImageDraw.Draw(img)

    if title:
        _draw_text_centered(draw, title, 40, title_font, _TITLE_COLOR, width)

    vi_y = height // 2 - 40
    en_y = height // 2 + 40

    if vi_text:
        _draw_text_centered(draw, vi_text, vi_y, vi_font, _VI_COLOR, width)
    if en_text:
        _draw_text_centered(draw, en_text, en_y, en_font, _EN_COLOR, width)

    return img.tobytes()


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------
def create_lyrics_video(
    audio_path: str,
    original_lrc_path: str,
    translated_lrc_path: str,
    output_video_path: str,
    title: str = "",
    artist: str = "",
    font_dir: str | None = None,
    video_width: int = 1920,
    video_height: int = 1080,
    fps: int = 2,
) -> str:
    """
    Create an MP4 lyrics video by rendering frames with Pillow and
    encoding via ffmpeg (piped raw frames — no libass needed).

    Uses a low FPS (default 2) since only text changes.
    Returns path to the created video.
    """
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Install: brew install ffmpeg")

    audio = Path(audio_path)
    if not audio.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    # --- Parse LRC files ---
    orig_events = parse_lrc(Path(original_lrc_path).read_text(encoding="utf-8"))
    trans_events = parse_lrc(Path(translated_lrc_path).read_text(encoding="utf-8"))

    if not orig_events:
        raise ValueError("No timed events in original LRC.")

    orig_ends = _compute_end_times(orig_events)
    trans_ends = _compute_end_times(trans_events)

    duration = max(orig_ends[-1] if orig_ends else 0, trans_ends[-1] if trans_ends else 0)
    duration = max(duration, 10.0)
    total_frames = int(math.ceil(duration * fps))

    # --- Load fonts ---
    if font_dir is None:
        project_fonts = Path(__file__).parent / "fonts" / "static"
        if project_fonts.is_dir():
            font_dir = str(project_fonts)

    vi_font = _load_font(font_dir, 52, bold=True)
    en_font = _load_font(font_dir, 40, bold=False)
    title_font = _load_font(font_dir, 32, bold=False)

    display_title = f"{title}  —  {artist}" if artist else title

    def _active_at(events, ends, t):
        for i, ev in enumerate(events):
            if ev["start_s"] <= t < ends[i]:
                return ev["text"]
        return ""

    # --- Start ffmpeg ---
    out_dir = Path(output_video_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{video_width}x{video_height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-i", str(audio),
        "-shortest",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_video_path),
    ]

    print(f"\n🎬  Generating lyrics video ({video_width}x{video_height} @ {fps} fps)")
    print(f"    Duration: ~{duration:.1f}s  ({total_frames} frames)")
    print(f"    Output:   {output_video_path}\n")

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    try:
        for frame_idx in range(total_frames):
            t = frame_idx / fps
            vi = _active_at(orig_events, orig_ends, t)
            en = _active_at(trans_events, trans_ends, t)

            raw = render_frame(
                video_width, video_height,
                vi, en, display_title,
                vi_font, en_font, title_font,
            )
            proc.stdin.write(raw)

            if frame_idx % (fps * 10) == 0:
                pct = frame_idx / total_frames * 100
                print(f"    [{pct:5.1f}%]  t={t:.1f}s", end="\r")

        proc.stdin.close()
        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            print(f"\n    ❌  ffmpeg error:\n{stderr.decode()}")
            raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

        print(f"\n    ✅  Video created: {output_video_path}")
        return str(output_video_path)

    except BrokenPipeError:
        stdout, stderr = proc.communicate()
        print(f"\n    ❌  ffmpeg pipe broken:\n{stderr.decode()}")
        raise RuntimeError("ffmpeg process terminated unexpectedly")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a lyrics video with Vietnamese + English subtitles."
    )
    parser.add_argument("audio", help="Path to the audio file")
    parser.add_argument(
        "-d", "--output-dir", default=None,
        help="Directory containing the LRC files (default: 'output/')",
    )
    parser.add_argument(
        "-o", "--output-video", default=None,
        help="Output video path (default: <output_dir>/lyrics_video.mp4)",
    )
    parser.add_argument("--title", default="", help="Song title for display")
    parser.add_argument("--artist", default="", help="Artist name for display")
    parser.add_argument("--width", type=int, default=1920, help="Video width")
    parser.add_argument("--height", type=int, default=1080, help="Video height")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second (default: 2)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path("output")
    orig_lrc = out_dir / "lyrics_original_synced.lrc"
    trans_lrc = out_dir / "lyrics_translated_synced.lrc"

    if not orig_lrc.exists():
        raise FileNotFoundError(f"Original synced lyrics not found: {orig_lrc}")
    if not trans_lrc.exists():
        raise FileNotFoundError(f"Translated synced lyrics not found: {trans_lrc}")

    output_video = args.output_video or str(out_dir / "lyrics_video.mp4")

    title, artist = args.title, args.artist
    meta_path = out_dir / "metadata.json"
    if meta_path.exists() and (not title or not artist):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        title = title or meta.get("title", "")
        artist = artist or meta.get("artist", "")

    create_lyrics_video(
        audio_path=str(args.audio),
        original_lrc_path=str(orig_lrc),
        translated_lrc_path=str(trans_lrc),
        output_video_path=output_video,
        title=title,
        artist=artist,
        video_width=args.width,
        video_height=args.height,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
