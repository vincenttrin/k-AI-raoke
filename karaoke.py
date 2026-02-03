import os
import subprocess
import whisperx # switched to whisperx for better timestamps
import torch

def create_karaoke_video(input_mp3, language_code, font_path):
    # Setup paths
    base_name = os.path.splitext(os.path.basename(input_mp3))[0]
    output_dir = os.path.join(os.getcwd(), f"{base_name}_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Separation (Same as before)
    print("--- Step 1: Separating Audio ---")
    subprocess.run([
        "demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", output_dir, input_mp3
    ], check=True)
    
    # Locate files
    demucs_song_dir = os.path.join(output_dir, "htdemucs", base_name)
    vocab_path = os.path.join(demucs_song_dir, "vocals.wav")
    inst_path = os.path.join(demucs_song_dir, "no_vocals.wav")

    # 2. Transcription with WhisperX (For precise timing)
    print(f"--- Step 2: Transcribing ({language_code}) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model (Use 'large-v2' or 'large-v3' for best Asian language results)
    model = whisperx.load_model("large-v2", device, compute_type="float16")
    audio = whisperx.load_audio(vocab_path)
    
    # Transcribe
    result = model.transcribe(audio, batch_size=16, language=language_code)
    
    # Align (This fixes the timing for Karaoke)
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # Save as SRT (Basic subtitle)
    # Note: For real karaoke with word-highlighting, you need to write a custom .ass generator
    # utilizing result['word_segments']. For now, we stick to SRT line-by-line.
    srt_path = os.path.join(output_dir, "lyrics.srt")
    # (Use the generate_srt function from the previous script here)
    # ... assuming generate_srt handles the WhisperX result structure ...
    
    # 3. Rendering with Custom Font
    print("--- Step 3: Rendering Video ---")
    output_video = os.path.join(output_dir, f"{base_name}_karaoke.mp4")
    
    # CRITICAL: Path formatting for FFmpeg
    # FFmpeg struggles with Windows paths in filter_complex, so we escape them carefully
    font_path_arg = font_path.replace("\\", "/").replace(":", "\\:")
    srt_path_arg = srt_path.replace("\\", "/").replace(":", "\\:")
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=30",
        "-i", inst_path,
        # ForceStyle sets the specific font file
        "-vf", f"subtitles='{srt_path_arg}':fontsdir='{os.path.dirname(font_path_arg)}':force_style='Fontname={os.path.splitext(os.path.basename(font_path))[0]},FontSize=24,PrimaryColour=&H00FFFF'",
        "-shortest",
        output_video
    ]
    
    subprocess.run(cmd, check=True)
    print(f"Done! Saved to {output_video}")

# Usage Example
# create_karaoke_video("kpop_song.mp3", "ko", "C:/Fonts/NotoSansKR-Bold.ttf")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create Karaoke Video from MP3")
    parser.add_argument("input_mp3", type=str, help="Path to input MP3 file")
    parser.add_argument("language_code", type=str, help="Language code for transcription (e.g., 'ko' for Korean)")
    parser.add_argument("font_path", type=str, help="Path to the TTF font file to use for subtitles")

    args = parser.parse_args()
    create_karaoke_video(args.input_mp3, args.language_code, args.font_path)

if __name__ == "__main__":
    main()