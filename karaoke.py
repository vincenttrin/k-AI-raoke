import os
import subprocess
import importlib.util
from pathlib import Path
import torch
import omegaconf # Ensure this is imported
import difflib
import re

# --- FIX FOR PYTORCH 2.6+ ---
# PyTorch 2.6+ blocks "omegaconf" by default. We must explicitly allow it.
try:
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata
    torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata])
    print("PyTorch 2.6+ fix applied: OmegaConf globals registered.")
except Exception as e:
    print(f"PyTorch fix warning: {e}")
    # Fallback: Attempt to force weights_only=False if the above fails
    try:
        torch.serialization.safe_globals = None 
    except: 
        pass
# -----------------------------

def reconcile_lyrics(whisper_segments, official_lyrics_text):
    """
    [DEPRECATED] Aligns official lyrics to WhisperX timestamps using sequence matching.
    This function is kept for backward compatibility but is no longer used.
    The script now uses forced alignment directly (see transcribe_vocals_to_ass).
    
    Returns a new list of segments with official text and borrowed timestamps.
    """
    # 1. Parse Official Text into words with line tracking
    official_words = []
    raw_lines = official_lyrics_text.strip().splitlines()
    
    for line_idx, line in enumerate(raw_lines):
        # clean punctuation for matching, but keep original for display if needed
        # simple whitespace split for now
        words_in_line = line.strip().split() 
        for w in words_in_line:
            official_words.append({
                "text": w,
                "line_idx": line_idx,
                "clean": re.sub(r'[^\w]', '', w.lower()) # normalized for comparison
            })

    # 2. Flatten Whisper segments into a single word list
    detected_words = []
    for seg in whisper_segments:
        if "words" in seg:
            for w in seg["words"]:
                detected_words.append({
                    "text": w["word"],
                    "start": w.get("start"),
                    "end": w.get("end"),
                    "clean": re.sub(r'[^\w]', '', w["word"].lower())
                })

    # 3. Match Sequences (The Magic Step)
    # We compare the "clean" versions of both lists
    matcher = difflib.SequenceMatcher(
        None, 
        [w["clean"] for w in detected_words], 
        [w["clean"] for w in official_words]
    )

    # 4. Transfer Timestamps
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ('equal', 'replace'):
            # Map detected timestamps to official words
            # We map the range of detected words to the range of official words
            det_slice = detected_words[i1:i2]
            off_slice = official_words[j1:j2]
            
            # Simple linear mapping if counts differ, otherwise 1:1
            for k, off_word in enumerate(off_slice):
                # Find best corresponding detected word index
                if not det_slice: continue
                det_idx = int((k / len(off_slice)) * len(det_slice))
                target_det = det_slice[det_idx]
                
                if off_word.get("start") is None: # Only set if not set
                    off_word["start"] = target_det["start"]
                    off_word["end"] = target_det["end"]

    # 5. Interpolate missing timestamps (for words that didn't match)
    # Fill gaps by looking at previous/next known times
    last_end = 0.0
    for w in official_words:
        if w.get("start") is None:
            w["start"] = last_end
        if w.get("end") is None:
            w["end"] = w["start"] + 0.5 # Default duration if completely missing
        last_end = w["end"]

    # 6. Re-group into Lines (Segments)
    new_segments = []
    current_line_idx = -1
    current_segment = None

    for w in official_words:
        if w["line_idx"] != current_line_idx:
            # New line detected, push previous segment
            if current_segment:
                new_segments.append(current_segment)
            
            # Start new segment
            current_line_idx = w["line_idx"]
            current_segment = {
                "text": "", 
                "start": w["start"], 
                "end": w["end"], 
                "words": []
            }
        
        # Add word to current segment
        # Add space if not first word
        sep = " " if current_segment["text"] else ""
        current_segment["text"] += sep + w["text"]
        current_segment["end"] = w["end"] # Extend segment end
        current_segment["words"].append(w)

    if current_segment:
        new_segments.append(current_segment)

    return new_segments

def _ensure_ctranslate2_rocm_stub():
    """Create the dummy ROCm DLL folder that ctranslate2 expects on Windows."""
    if os.name != "nt":
        return

    try:
        spec = importlib.util.find_spec("ctranslate2")
        if not spec or not spec.origin:
            return

        package_dir = Path(spec.origin).resolve().parent
        for folder in ("_rocm_sdk_core", "_rocm_sdk_libraries_custom"):
            rocm_bin = (package_dir / ".." / folder / "bin").resolve()
            rocm_bin.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Warning: unable to prepare ctranslate2 DLL directory: {exc}")

def separate_audio_sources(input_mp3):
    """Run Demucs separation and return relevant file paths."""
    base_name = os.path.splitext(os.path.basename(input_mp3))[0]
    output_dir = os.path.join(os.getcwd(), f"{base_name}_output")
    os.makedirs(output_dir, exist_ok=True)

    print("--- Step 1: Separating Audio ---")
    subprocess.run([
        "demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", output_dir, input_mp3
    ], check=True)

    demucs_song_dir = os.path.join(output_dir, "htdemucs", base_name)
    vocals_path = os.path.join(demucs_song_dir, "vocals.wav")
    inst_path = os.path.join(demucs_song_dir, "no_vocals.wav")

    return {
        "base_name": base_name,
        "output_dir": output_dir,
        "demucs_song_dir": demucs_song_dir,
        "vocals_path": vocals_path,
        "instrumental_path": inst_path,
    }


def transcribe_vocals_to_ass(vocals_path, language_code, output_dir, official_lyrics_path=None):
    """Use WhisperX to align or transcribe the vocals and return the planned ASS path."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ensure_ctranslate2_rocm_stub()
    import whisperx 
    
    audio = whisperx.load_audio(vocals_path)
    
    # Load alignment model (needed for both paths)
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    
    # --- FORCED ALIGNMENT PATH (if official lyrics provided) ---
    if official_lyrics_path and os.path.exists(official_lyrics_path):
        print(f"--- Step 2: Forced Alignment with Official Lyrics ({language_code}) ---")
        try:
            with open(official_lyrics_path, "r", encoding="utf-8") as f:
                official_text = f.read()
            
            # Convert lyrics to segments format expected by WhisperX align
            # Split into lines, treating each line as a segment
            lines = [line.strip() for line in official_text.strip().splitlines() if line.strip()]
            
            # Create pseudo-segments (WhisperX align expects this format)
            # We'll estimate timing based on audio length divided by number of lines
            audio_duration = len(audio) / 16000.0  # WhisperX loads at 16kHz
            estimated_duration_per_line = audio_duration / len(lines) if lines else 1.0
            
            pseudo_segments = []
            for i, line in enumerate(lines):
                pseudo_segments.append({
                    "start": i * estimated_duration_per_line,
                    "end": (i + 1) * estimated_duration_per_line,
                    "text": line
                })
            
            # Use WhisperX align to force-align the official lyrics
            print("Running forced alignment...")
            aligned_result = whisperx.align(
                pseudo_segments, 
                model_a, 
                metadata, 
                audio, 
                device, 
                return_char_alignments=False
            )
            
            final_segments = aligned_result["segments"]
            print("Forced alignment successful.")
            
        except Exception as e:
            print(f"Warning: Forced alignment failed ({e}). Falling back to transcription.")
            official_lyrics_path = None  # Force fallback
    
    # --- TRANSCRIPTION PATH (fallback or default) ---
    if not official_lyrics_path or not os.path.exists(official_lyrics_path):
        print(f"--- Step 2: Transcribing ({language_code}) ---")
        compute_type = "float16" if device == "cuda" else "float32"
        model = whisperx.load_model("large-v2", device=device, compute_type=compute_type)
        
        result = model.transcribe(audio, batch_size=16, language=language_code)
        aligned_result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            device, 
            return_char_alignments=False
        )
        
        final_segments = aligned_result["segments"]
        print("Transcription complete.")
    
    # --- GENERATE ASS FILE ---
    resolved_output_dir = output_dir or os.path.dirname(os.path.abspath(vocals_path))
    os.makedirs(resolved_output_dir, exist_ok=True)
    
    ass_path = os.path.join(resolved_output_dir, "lyrics.ass")
    generate_karaoke_ass(final_segments, ass_path, font_name="Arial") 
    
    return {
        "aligned_segments": final_segments,
        "srt_path": ass_path,  # Return the ASS path here
    }


def format_ass_time(seconds):
    """Convert seconds to ASS format h:mm:ss.cc"""
    if seconds is None: seconds = 0.0
    cs = int(round(seconds * 100)) # centiseconds
    h, r = divmod(cs, 360000)
    m, r = divmod(r, 6000)
    s, cs = divmod(r, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def generate_karaoke_ass(aligned_result, ass_path, font_name="Arial"):
    """
    Generate an .ass file with karaoke tags based on word-level timestamps.
    """
    segments = aligned_result.get("segments") if isinstance(aligned_result, dict) else aligned_result
    
    # ASS Header
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{font_name},60,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,1,0,1,3,0,5,10,10,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        
        for seg in segments:
            if "words" not in seg:
                continue

            start_time = seg["start"]
            end_time = seg["end"]
            
            # Build the karaoke line: {\k20}Word {\k30}Word
            karaoke_line = ""
            current_time = start_time
            
            for idx, word in enumerate(seg["words"]):
                w_start = word.get("start", current_time)
                w_end = word.get("end", w_start + 0.1)
                text = word["word"]
                
                # Use the actual word duration for accurate highlighting
                duration = w_end - w_start
                k_duration = int(duration * 100)  # Convert to centiseconds
                
                # Add space to text if strictly needed, though usually handled by renderer
                karaoke_line += f"{{\\k{k_duration}}}{text} "
                current_time = w_end

            # Write the dialogue line
            f.write(f"Dialogue: 0,{format_ass_time(start_time)},{format_ass_time(end_time)},Karaoke,,0,0,0,,{karaoke_line}\n")

def render_karaoke_output(inst_path, subtitle_path, font_path, output_dir, base_name):
    """Render the final karaoke video with FFmpeg."""
    print("--- Step 3: Rendering Video ---")
    output_video = os.path.join(output_dir, f"{base_name}_karaoke.mp4")

    font_path_arg = font_path.replace("\\", "/").replace(":", "\\:")
    subtitle_path_arg = subtitle_path.replace("\\", "/").replace(":", "\\:")

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=30",
        "-i", inst_path,
        "-vf", f"ass='{subtitle_path_arg}':fontsdir='{os.path.dirname(font_path_arg)}'",
        "-shortest",
        output_video
    ]

    subprocess.run(cmd, check=True)
    print(f"Done! Saved to {output_video}")

    return output_video


def create_karaoke_video(input_mp3, language_code, font_path):
    """High-level convenience wrapper that runs all three stages."""
    separation_artifacts = separate_audio_sources(input_mp3)
    transcription_artifacts = transcribe_vocals_to_ass(
        separation_artifacts["vocals_path"], language_code, separation_artifacts["output_dir"]
    )
    return render_karaoke_output(
        separation_artifacts["instrumental_path"],
        transcription_artifacts["srt_path"],
        font_path,
        separation_artifacts["output_dir"],
        separation_artifacts["base_name"],
    )

# Usage Example
# create_karaoke_video("kpop_song.mp3", "ko", "C:/Fonts/NotoSansKR-Bold.ttf")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create Karaoke Video from MP3")
    parser.add_argument("input_mp3", type=str, help="Path to input MP3 file")
    parser.add_argument("language_code", type=str, help="Language code for transcription (e.g., 'ko' for Korean)")
    parser.add_argument("font_path", type=str, help="Path to the TTF font file to use for subtitles")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (optional)")
    parser.add_argument("--official_lyrics", type=str, default=None, help="Path to official lyrics text file (optional)")
    
    args = parser.parse_args()
    # create_karaoke_video(args.input_mp3, args.language_code, args.font_path)
    # print(args.output_dir)
    # print(type(args.output_dir))
    # transcribe_vocals_to_ass(args.input_mp3, args.language_code, args.output_dir, args.official_lyrics)\
    render_karaoke_output(r"who-knows-DC_output\htdemucs\who-knows-DC\no_vocals.wav", r"who-knows-DC_output\lyrics.ass", args.font_path, "who-knows-DC_output", "who_knows")

if __name__ == "__main__":
    main()