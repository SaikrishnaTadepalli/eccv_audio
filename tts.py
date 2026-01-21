from gtts import gTTS
import os

def text_to_mp3(input_text, output_filename):
  if not output_filename.endswith('.mp3'):
    output_filename += '.mp3'

  try:
    print(f"Generating audio for: '{output_filename}' ...")

    tts = gTTS(text=input_text, lang='en', slow=False)
    tts.save(output_filename)

    print(f"Saved audio file to '{os.path.abspath(output_filename)}'.")

  except Exception as e:
    print(f"Something wrong: {e}")


if __name__ == '__main__':
  text_to_mp3(
    "Hello, World! This is an audio file.",
    "test_audio.mp3"
  )
