import os
from pathlib import Path
from gtts import gTTS

# MODIFY OUTPUT DIRECTORY HERE
OUTPUT_DIR = Path("./outputs/")


def init_output_dir():
    """Ensures the output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_unique_filename(base_name: str) -> str:
    """
    Checks for file existence and returns a unique filename string.
    Example: 'test' -> 'test_v1' if 'test.mp3' exists.
    """
    stem = Path(base_name).stem
    candidate_stem = stem
    counter = 1
    
    # Check if a file with this name (either .mp3 or .txt) already exists
    while (OUTPUT_DIR / f"{candidate_stem}.mp3").exists() or \
          (OUTPUT_DIR / f"{candidate_stem}.txt").exists():
        candidate_stem = f"{stem}_v{counter}"
        counter += 1
    
    return candidate_stem


def text_to_mp3(input_text: str, output_filename: str = "file_save"):
    """Converts text to an MP3 and saves a transcript copy."""
    if not input_text.strip():
        print("Error: Input text is empty.")
        return

    # Creates output directory if it doesn't exist
    init_output_dir()
    
    # Generate a safe, unique filename
    unique_stem = get_unique_filename(output_filename)

    audio_path = OUTPUT_DIR / f"{unique_stem}.mp3"
    text_path = OUTPUT_DIR / f"{unique_stem}.txt"

    try:
        print(f"Processing: '{unique_stem}'...")

        # Generate and save audio
        tts = gTTS(text=input_text, lang='en', slow=False)
        tts.save(str(audio_path))

        # Save transcript
        text_path.write_text(input_text, encoding="utf-8")

        print(f"\t[✓] Transcript: {text_path.absolute()}")
        print(f"\t[✓] Audio:      {audio_path.absolute()}\n")

    except Exception as e:
        print(f"Failed to process {unique_stem}: {e}")


def batch_convert(input_texts, file_names=None):
    """Batch convert list of texts to audio files."""
    if file_names:
        assert(len(input_texts) == len(file_names))
        for input_text, file_name in zip(input_texts, file_names):
            text_to_mp3(input_text, file_name)
    else:
        for input_text in input_texts:
            text_to_mp3(input_text)


if __name__ == '__main__':
    text_to_mp3(
        """Hello, World! This is an audio file. Now I will read some random text.
        Paragraphs are the building blocks of papers. Many students define 
        paragraphs in terms of length: a paragraph is a group of at least five 
        sentences, a paragraph is half a page long, etc. In reality, though, the 
        unity and coherence of ideas among sentences is what constitutes a paragraph. 
        A paragraph is defined as “a group of sentences or a single sentence that 
        forms a unit” (Lunsford and Connors 116). Length and appearance do not 
        determine whether a section in a paper is a paragraph. For instance, in some 
        styles of writing, particularly journalistic styles, a paragraph can be just 
        one sentence long. Ultimately, a paragraph is a sentence or group of sentences 
        that support one main idea. In this handout, we will refer to this as the 
        “controlling idea,” because it controls what happens in the rest of the paragraph.
        """, 
        "input"
    )
    # Single Convert
    # text_to_mp3("Hello! This is a single test.", "welcome")

    # # Batch Convert
    # messages = ["First version", "Second version", "Third version"]
    # for msg in messages:
    #     text_to_mp3(msg, "duplicate_name_test")

