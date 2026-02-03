import os
import subprocess
import importlib.util
from pathlib import Path
import torch


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


def transcribe_vocals_to_srt(vocals_path, language_code, output_dir):
    """Use WhisperX to transcribe the vocals and return the planned SRT path."""
    print(f"--- Step 2: Transcribing ({language_code}) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ensure_ctranslate2_rocm_stub()
    import whisperx  # switched to whisperx for better timestamps

    compute_type = "float16" if device == "cuda" else "float32"
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    audio = whisperx.load_audio(vocals_path)

    result = model.transcribe(audio, batch_size=16, language=language_code)
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    srt_path = os.path.join(output_dir, "lyrics.srt")
    # (Use the generate_srt function from the previous script here)
    # ... assuming generate_srt handles the WhisperX result structure ...

    return {
        "aligned_segments": aligned_result,
        "srt_path": srt_path,
    }


def render_karaoke_output(inst_path, srt_path, font_path, output_dir, base_name):
    """Render the final karaoke video with FFmpeg."""
    print("--- Step 3: Rendering Video ---")
    output_video = os.path.join(output_dir, f"{base_name}_karaoke.mp4")

    font_path_arg = font_path.replace("\\", "/").replace(":", "\\:")
    srt_path_arg = srt_path.replace("\\", "/").replace(":", "\\:")

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=30",
        "-i", inst_path,
        "-vf", f"subtitles='{srt_path_arg}':fontsdir='{os.path.dirname(font_path_arg)}':force_style='Fontname={os.path.splitext(os.path.basename(font_path))[0]},FontSize=24,PrimaryColour=&H00FFFF'",
        "-shortest",
        output_video
    ]

    subprocess.run(cmd, check=True)
    print(f"Done! Saved to {output_video}")

    return output_video


def create_karaoke_video(input_mp3, language_code, font_path):
    """High-level convenience wrapper that runs all three stages."""
    separation_artifacts = separate_audio_sources(input_mp3)
    transcription_artifacts = transcribe_vocals_to_srt(
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
    args = parser.parse_args()
    vocal_generate_srt(args.input_mp3, args.language_code, 'who-knows-DC_output/lyrics.srt')
    # create_karaoke_video(args.input_mp3, args.language_code, args.font_path)

if __name__ == "__main__":
    main()