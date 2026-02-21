"""
translate_lyrics.py
~~~~~~~~~~~~~~~~~~~
Pipeline:  Vietnamese audio file  →  song identification (metadata / manual)
           →  lyrics retrieval (LRCLib)  →  Vietnamese→English translation

Dependencies: tinytag, requests, deep-translator
"""

import json
import re
import subprocess
import textwrap
from pathlib import Path
from urllib.parse import urlencode

from deep_translator import GoogleTranslator
from tinytag import TinyTag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LRCLIB_BASE = "https://lrclib.net/api"
LRCLIB_HEADERS = {
    "User-Agent": "k-AI-raoke v1.0 (https://github.com/tamyboi/k-AI-raoke)"
}

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}


# ---------------------------------------------------------------------------
# 1.  Song identification via audio-file metadata (ID3 / Vorbis / etc.)
# ---------------------------------------------------------------------------
def identify_song(
    audio_path: str,
    title_override: str | None = None,
    artist_override: str | None = None,
) -> dict:
    """
    Read song metadata from an audio file using TinyTag.

    If *title_override* or *artist_override* are provided they take
    precedence over whatever is embedded in the file.

    Returns a dict with keys: title, artist, album, duration
    Raises RuntimeError if no title can be determined.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    print(f"\n🎵  Reading metadata from: {path.name} …")
    tag = TinyTag.get(str(path))

    title = title_override or tag.title or ""
    artist = artist_override or tag.artist or ""
    album = tag.album or ""
    duration = int(tag.duration or 0)

    # Fallback: derive title from filename
    if not title:
        title = path.stem  # e.g. "ANH LAM GI SAI - CHAU KHAI PHONG"
        print(f"    ⚠️  No title tag found — using filename: {title}")

    if not artist:
        # Try to split "TITLE - ARTIST" pattern from filename
        if " - " in title:
            parts = title.split(" - ", 1)
            title, artist = parts[0].strip(), parts[1].strip()
            print(f"    ⚠️  Guessed artist from filename: {artist}")

    if not title:
        raise RuntimeError(
            "Cannot determine song title. "
            "Please provide --title and --artist on the command line."
        )

    print(f"    ✅  Title : {title}")
    print(f"    🎤  Artist: {artist or '(unknown)'}")
    if album:
        print(f"    💿  Album : {album}")
    if duration:
        print(f"    ⏱️  Duration: {duration}s")

    return {
        "title": title,
        "artist": artist,
        "album": album,
        "duration": duration,
    }


# ---------------------------------------------------------------------------
# 2.  Lyrics retrieval via LRCLib
# ---------------------------------------------------------------------------
def _lrclib_get(endpoint: str, params: dict) -> list | dict:
    """
    Call LRCLib API using curl to avoid Python 3.14 SSL handshake issues.
    Retries up to 3 times on transient failures.
    Returns parsed JSON (list or dict).
    """
    url = f"{LRCLIB_BASE}/{endpoint}?{urlencode(params)}"
    user_agent = LRCLIB_HEADERS["User-Agent"]
    cmd = [
        "curl", "-sS",
        "--tlsv1.2",           # force TLS 1.2 for compatibility
        "--retry", "3",        # curl-level retries on transient errors
        "--retry-delay", "2",
        "--max-time", "20",
        "-H", f"User-Agent: {user_agent}",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed (exit {result.returncode}): {result.stderr.strip()}")
    body = result.stdout.strip()
    if not body:
        return []
    return json.loads(body)


def search_lyrics(track_name: str, artist_name: str) -> list[dict]:
    """
    Search LRCLib for lyrics matching the given track and artist.
    Returns a list of result dicts (may be empty).
    """
    data = _lrclib_get("search", {"track_name": track_name, "artist_name": artist_name})
    return data if isinstance(data, list) else []


def search_lyrics_query(query: str) -> list[dict]:
    """
    Fallback: search LRCLib with a free-text query.
    """
    data = _lrclib_get("search", {"q": query})
    return data if isinstance(data, list) else []


def fetch_lyrics(track_name: str, artist_name: str) -> dict | None:
    """
    Fetch lyrics for a track.  Tries a structured search first, then
    falls back to a free-text query.

    Returns the best-matching record dict (with plainLyrics / syncedLyrics)
    or None if nothing is found.
    """
    print(f"\n🔍  Searching LRCLib for: {track_name} — {artist_name} ...")

    # --- Attempt 1: structured search ---
    results = search_lyrics(track_name, artist_name)

    # --- Attempt 2: broader free-text query ---
    if not results:
        print("    ⚠️  No structured match — trying free-text search …")
        results = search_lyrics_query(f"{track_name} {artist_name}")

    # --- Attempt 3: track name only ---
    if not results:
        print("    ⚠️  No results — trying track name only …")
        results = search_lyrics_query(track_name)

    if not results:
        print("    ❌  No lyrics found on LRCLib.")
        return None

    # Pick the first result that actually has lyrics text
    for r in results:
        if r.get("plainLyrics") or r.get("syncedLyrics"):
            print(
                f"    ✅  Found lyrics: {r.get('trackName')} — "
                f"{r.get('artistName')} (id={r.get('id')})"
            )
            return r

    print("    ❌  Results returned but none contained lyrics text.")
    return None


# ---------------------------------------------------------------------------
# 3.  Translation (Vietnamese → English)
# ---------------------------------------------------------------------------
def _chunk_text(text: str, max_chars: int = 4500) -> list[str]:
    """
    Split text into chunks that respect line boundaries and stay under
    the Google Translate per-request character limit (~5 000 chars).
    """
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current = ""
    for line in lines:
        if len(current) + len(line) > max_chars:
            chunks.append(current)
            current = line
        else:
            current += line
    if current:
        chunks.append(current)
    return chunks


def translate_text(text: str, source: str = "vi", target: str = "en") -> str:
    """
    Translate *text* from *source* language to *target* language using
    Google Translate (via deep-translator).  Handles long texts by
    chunking automatically.
    """
    if not text or not text.strip():
        return ""

    translator = GoogleTranslator(source=source, target=target)
    chunks = _chunk_text(text)
    translated_chunks: list[str] = []

    for i, chunk in enumerate(chunks, 1):
        translated = translator.translate(chunk)
        translated_chunks.append(translated)

    return "\n".join(translated_chunks)


def translate_synced_lyrics(synced_lyrics: str, source: str = "vi", target: str = "en") -> str:
    """
    Translate synced (LRC-format) lyrics while preserving timestamps.
    Input format:  [mm:ss.xx] Line of lyrics
    """
    if not synced_lyrics or not synced_lyrics.strip():
        return ""

    timestamp_re = re.compile(r"^(\[\d{2}:\d{2}\.\d{2,3}\])\s*(.*)")
    lines = synced_lyrics.strip().splitlines()

    timestamps: list[str] = []
    text_lines: list[str] = []

    for line in lines:
        m = timestamp_re.match(line)
        if m:
            timestamps.append(m.group(1))
            text_lines.append(m.group(2))
        else:
            timestamps.append("")
            text_lines.append(line)

    # Translate all text lines at once (joined) then split back
    joined = "\n".join(text_lines)
    translated_joined = translate_text(joined, source=source, target=target)
    translated_lines = translated_joined.splitlines()

    # Re-attach timestamps
    result_lines: list[str] = []
    for i, ts in enumerate(timestamps):
        t_line = translated_lines[i] if i < len(translated_lines) else ""
        if ts:
            result_lines.append(f"{ts} {t_line}")
        else:
            result_lines.append(t_line)

    return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# 4.  Orchestrator
# ---------------------------------------------------------------------------
def process_audio(
    audio_path: str,
    output_dir: str | None = None,
    source_lang: str = "vi",
    target_lang: str = "en",
    title_override: str | None = None,
    artist_override: str | None = None,
) -> dict:
    """
    End-to-end pipeline:
      1. Identify the song from file metadata (or CLI overrides)
      2. Fetch lyrics from LRCLib
      3. Translate lyrics Vietnamese → English
      4. Save outputs to *output_dir*

    Returns a summary dict.
    """
    audio_path = str(Path(audio_path).resolve())
    song = identify_song(audio_path, title_override, artist_override)

    record = fetch_lyrics(song["title"], song["artist"])

    if record is None:
        return {
            "status": "no_lyrics",
            "song": song,
            "message": "Song identified but no lyrics found on LRCLib.",
        }

    plain = record.get("plainLyrics") or ""
    synced = record.get("syncedLyrics") or ""

    # --- Translate ---
    print(f"\n🌐  Translating lyrics ({source_lang} → {target_lang}) …")
    plain_translated = ""
    synced_translated = ""

    if plain:
        plain_translated = translate_text(plain, source=source_lang, target=target_lang)
        print("    ✅  Plain lyrics translated.")

    if synced:
        synced_translated = translate_synced_lyrics(synced, source=source_lang, target=target_lang)
        print("    ✅  Synced lyrics translated.")

    # --- Save outputs ---
    if output_dir is None:
        stem = Path(audio_path).stem
        output_dir = str(Path(audio_path).parent / f"{stem}_translated")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _write(name: str, content: str) -> Path | None:
        if not content:
            return None
        p = out / name
        p.write_text(content, encoding="utf-8")
        return p

    saved = {}
    saved["original_plain"] = _write("lyrics_original.txt", plain)
    saved["original_synced"] = _write("lyrics_original_synced.lrc", synced)
    saved["translated_plain"] = _write("lyrics_translated.txt", plain_translated)
    saved["translated_synced"] = _write("lyrics_translated_synced.lrc", synced_translated)

    # Also save a combined side-by-side view
    if plain and plain_translated:
        side_by_side = _build_side_by_side(plain, plain_translated)
        saved["side_by_side"] = _write("lyrics_side_by_side.txt", side_by_side)

    # Save metadata
    meta = {
        "title": song["title"],
        "artist": song["artist"],
        "album": song.get("album", ""),
        "lrclib_id": record.get("id"),
        "source_lang": source_lang,
        "target_lang": target_lang,
    }
    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    saved["metadata"] = meta_path

    print(f"\n📁  Output saved to: {out}")
    for label, p in saved.items():
        if p:
            print(f"    • {label}: {Path(p).name}")

    return {
        "status": "ok",
        "song": song,
        "original_lyrics": plain,
        "translated_lyrics": plain_translated,
        "original_synced": synced,
        "translated_synced": synced_translated,
        "output_dir": str(out),
        "files": {k: str(v) for k, v in saved.items() if v},
    }


def _build_side_by_side(original: str, translated: str) -> str:
    """Build a human-readable side-by-side comparison."""
    orig_lines = original.strip().splitlines()
    trans_lines = translated.strip().splitlines()

    width = 50
    header = (
        f"{'ORIGINAL (Vietnamese)':<{width}} | {'TRANSLATED (English)'}\n"
        f"{'-' * width}-+-{'-' * width}\n"
    )

    lines = []
    for i in range(max(len(orig_lines), len(trans_lines))):
        o = orig_lines[i] if i < len(orig_lines) else ""
        t = trans_lines[i] if i < len(trans_lines) else ""
        lines.append(f"{o:<{width}} | {t}")

    return header + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI helper  (also usable standalone: python translate_lyrics.py <file>)
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Identify a Vietnamese song, fetch lyrics, and translate to English."
    )
    parser.add_argument("audio", help="Path to the audio file (mp3, wav, m4a, …)")
    parser.add_argument(
        "-o", "--output", default=None, help="Output directory (default: <audio>_translated/)"
    )
    parser.add_argument(
        "--title", default=None,
        help="Song title (overrides file metadata)",
    )
    parser.add_argument(
        "--artist", default=None,
        help="Artist name (overrides file metadata)",
    )
    parser.add_argument(
        "--source-lang", default="vi", help="Source language code (default: vi)"
    )
    parser.add_argument(
        "--target-lang", default="en", help="Target language code (default: en)"
    )
    parser.add_argument(
        "--video", action="store_true",
        help="Also generate a lyrics video (Vietnamese + English subtitles)",
    )
    parser.add_argument(
        "--fps", type=int, default=2,
        help="Video frames per second (default: 2, only used with --video)",
    )
    args = parser.parse_args()

    result = process_audio(
        args.audio,
        output_dir=args.output,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        title_override=args.title,
        artist_override=args.artist,
    )

    if result["status"] == "no_lyrics":
        print(f"\n⚠️  {result['message']}")
        print(f"    Song: {result['song']['title']} — {result['song']['artist']}")
        return

    # Print translated lyrics to stdout
    print("\n" + "=" * 60)
    print("  TRANSLATED LYRICS")
    print("=" * 60)
    print(result["translated_lyrics"])
    print("=" * 60)

    # Optionally generate video
    if args.video:
        from generate_video import create_lyrics_video

        out_dir = Path(result["output_dir"])
        orig_lrc = out_dir / "lyrics_original_synced.lrc"
        trans_lrc = out_dir / "lyrics_translated_synced.lrc"

        if not orig_lrc.exists() or not trans_lrc.exists():
            print("\n⚠️  Cannot generate video — synced lyrics files are missing.")
            return

        video_path = str(out_dir / "lyrics_video.mp4")
        create_lyrics_video(
            audio_path=args.audio,
            original_lrc_path=str(orig_lrc),
            translated_lrc_path=str(trans_lrc),
            output_video_path=video_path,
            title=result["song"]["title"],
            artist=result["song"]["artist"],
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
