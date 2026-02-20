#!/usr/bin/env python3
"""
main.py — k-AI-raoke: Vietnamese Song Lyrics Translator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Takes a Vietnamese audio file, identifies the song via Shazam,
fetches lyrics from LRCLib, and translates them to English.

Usage:
    python main.py <audio_file>
    python main.py songs_input/your_song.mp3
    python main.py songs_input/your_song.mp3 -o output_folder/
    python main.py songs_input/your_song.mp3 --source-lang vi --target-lang en
"""

from translate_lyrics import main

if __name__ == "__main__":
    main()
