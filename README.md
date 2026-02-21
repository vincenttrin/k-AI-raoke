# k-AI-raoke 🎤

**Turn any Vietnamese song into a karaoke experience — with English translations.**

k-AI-raoke takes an audio file, automatically identifies the song, fetches its lyrics, translates them from Vietnamese to English, and generates a karaoke-style video with both languages displayed on screen.

---

## What Does It Do?

Give the app a Vietnamese song file (MP3, WAV, M4A, etc.) and it will:

1. **Read the song's metadata** (title, artist) from the file — or accept them manually.
2. **Fetch the official lyrics** (including time-synced lyrics) from [LRCLib](https://lrclib.net).
3. **Translate the lyrics** from Vietnamese to English using Google Translate.
4. **Generate a lyrics video** (MP4) showing the original Vietnamese lyrics in white and the English translation in yellow, synced to the music.

All output files are saved to a folder you choose (or a default `output/` folder).

### Output Files

| File | Description |
|------|-------------|
| `lyrics_original.txt` | Plain Vietnamese lyrics |
| `lyrics_translated.txt` | Plain English translation |
| `lyrics_original_synced.lrc` | Vietnamese lyrics with timestamps (LRC format) |
| `lyrics_translated_synced.lrc` | English lyrics with timestamps (LRC format) |
| `lyrics_side_by_side.txt` | Side-by-side Vietnamese / English comparison |
| `metadata.json` | Song metadata (title, artist, album, etc.) |
| `lyrics_video.mp4` | Karaoke-style video with bilingual subtitles *(only with `--video` flag)* |

---

## Prerequisites

Before setting up the app, make sure you have the following installed on your computer. If you don't have them yet, follow the instructions below for your system.

### 1. Python 3.10 or newer

Check if Python is already installed by opening **Terminal** (Mac) or **Command Prompt** (Windows) and typing:

```bash
python3 --version
```

If you see a version number like `Python 3.10.x` or higher, you're good. Otherwise:

- **Mac:** Install from [python.org](https://www.python.org/downloads/) or via Homebrew:
  ```bash
  brew install python
  ```
- **Windows:** Download and install from [python.org](https://www.python.org/downloads/). **Make sure to check "Add Python to PATH"** during installation.

### 2. ffmpeg

ffmpeg is a free tool used to create the video. Check if it's installed:

```bash
ffmpeg -version
```

If not installed:

- **Mac:**
  ```bash
  brew install ffmpeg
  ```
- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html), extract it, and add the `bin` folder to your system PATH.

### 3. curl

curl is used to fetch lyrics from the internet. It comes pre-installed on Mac and most Linux systems. On Windows 10+, it's also built in. Verify with:

```bash
curl --version
```

---

## Setup (First-Time Only)

Follow these steps once to get everything ready.

### Step 1: Download the Project

If you received the project as a ZIP file, unzip it to a location you'll remember (e.g., your Desktop or Documents folder).

If you have Git installed, you can clone it instead:

```bash
git clone https://github.com/tamyboi/k-AI-raoke.git
```

### Step 2: Open a Terminal in the Project Folder

Open **Terminal** (Mac) or **Command Prompt / PowerShell** (Windows) and navigate to the project folder:

```bash
cd /path/to/k-AI-raoke
```

> **Tip:** On Mac, you can type `cd ` (with a space) and then drag the folder from Finder into the Terminal window.

### Step 3: Create a Virtual Environment

A virtual environment keeps this project's packages separate from the rest of your system.

```bash
python3 -m venv .venv
```

### Step 4: Activate the Virtual Environment

- **Mac / Linux:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate
  ```
- **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

When the environment is active you'll see `(.venv)` at the beginning of your terminal prompt.

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs the Python libraries the app needs (tinytag, deep-translator, Pillow, etc.).

> **Note:** The first run may also download AI models (WhisperX, Demucs) which can be several hundred MB. This only happens once.

---

## How to Run

Make sure you're in the project folder and the virtual environment is activated (you should see `(.venv)` in your prompt — if not, run the activate command from Step 4 above).

### Basic Usage — Translate Lyrics Only

```bash
python3 main.py "songs_input/your_song.mp3" --title "Song Title" --artist "Artist Name" -o output
```

This will:
- Identify the song (using the title/artist you provide, or from the file's metadata).
- Fetch and translate the lyrics.
- Save all text files to the `output/` folder.

### Generate a Karaoke Video

Add the `--video` flag to also create an MP4 video:

```bash
python3 main.py "songs_input/your_song.mp3" --title "Song Title" --artist "Artist Name" -o output --video
```

The video will be saved as `output/lyrics_video.mp4`.

### Full List of Options

| Option | Description | Default |
|--------|-------------|---------|
| `audio` | Path to your audio file (required) | — |
| `--title` | Song title (overrides file metadata) | Read from file |
| `--artist` | Artist name (overrides file metadata) | Read from file |
| `-o`, `--output` | Output folder for all generated files | `<filename>_translated/` |
| `--source-lang` | Source language code | `vi` (Vietnamese) |
| `--target-lang` | Target language code | `en` (English) |
| `--video` | Also generate a lyrics video | Off |
| `--fps` | Video frames per second (only with `--video`) | `2` |

### Example

```bash
python3 main.py "songs_input/Timeline 1.wav" --title "Anh Lam Gi Sai" --artist "Chau Khai Phong" -o output --video
```

---

## Project Structure

```
k-AI-raoke/
├── main.py                 # Entry point — run this
├── translate_lyrics.py     # Song identification, lyrics fetching & translation
├── generate_video.py       # Lyrics video generation (Pillow + ffmpeg)
├── karaoke.py              # Audio separation (Demucs) & vocal transcription (WhisperX)
├── requirements.txt        # Python dependencies
├── fonts/
│   └── static/             # Bundled Noto Sans fonts
├── songs_input/            # Place your audio files here
├── output/                 # Generated lyrics & video files
└── README.md               # You are here
```

---

## Troubleshooting

### "No lyrics found on LRCLib"

LRCLib may not have the song in its database. Try:
- Double-checking the `--title` and `--artist` spelling.
- Searching [lrclib.net](https://lrclib.net) manually to see if the song exists.

### "ffmpeg not found"

Make sure ffmpeg is installed and available on your system PATH (see [Prerequisites](#2-ffmpeg) above).

### "python3: command not found"

- On Windows, try `python` instead of `python3`.
- Make sure Python is installed and added to your PATH.

### Slow first run

The first time you run the app it may need to download AI models (several hundred MB). Subsequent runs will be much faster.

### Virtual environment issues

If you see permission errors or package-not-found errors, make sure the virtual environment is activated (you should see `(.venv)` in your prompt). Re-run:

```bash
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

---

## License

Fonts included in `fonts/` are licensed under the [SIL Open Font License](fonts/OFL.txt).
