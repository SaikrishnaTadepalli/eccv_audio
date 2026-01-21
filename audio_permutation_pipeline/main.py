import librosa
import soundfile as sf
import numpy as np
from scipy.signal import fftconvolve
from pydub import AudioSegment
import os
import shutil
from pathlib import Path

# MODIFY OUTPUT DIRECTORY HERE
OUTPUT_DIR = Path("./outputs/")


def init_output_dir():
    """Ensures the output directory exists."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def shift_pitch(data, sr, n_steps=0):
    """
    Shifts pitch (without changing duration).
    - positive 'n_steps' increases pitch
    - negative 'n_steps' decreases pitch 
    """
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)


def stretch_time(data, rate=1.0):
    """
    Changes speed (without changing pitch).
    - 'rate' > 1.0 increases speed
    - 'rate' < 1.0 decreases speed
    """
    return librosa.effects.time_stretch(y=data, rate=rate)


def apply_reverb(data, sr, room_size=0.5, wet_dry=0.3):
    """
    Adds reverb effect to audio.
    - 'room_size' controls the reverb length (0.0 to 1.0)
    - 'wet_dry' mixes between original and reverb (0.0 = dry, 1.0 = wet)
    """
    # Create impulse response for reverb
    reverb_duration = room_size * 2.0  # seconds
    ir_length = int(reverb_duration * sr)
    
    # Generate decaying noise impulse response
    t = np.linspace(0, reverb_duration, ir_length)
    decay = np.exp(-3.0 * t / reverb_duration)
    impulse = decay * np.random.randn(ir_length) * 0.1
    
    # Convolve audio with impulse response
    reverb_signal = fftconvolve(data, impulse, mode='same')
    
    # Mix dry and wet signals
    output = (1 - wet_dry) * data + wet_dry * reverb_signal
    
    # Normalize to prevent clipping
    output = output / np.max(np.abs(output))
    
    return output


def save_as_mp3(data, sr, output_path):
    """
    Save audio data as MP3 file.
    - librosa only works with WAV
    - need to convert WAVs back to MP3
    """
    # Save to temporary WAV file
    temp_wav = output_path.replace('.mp3', '_temp.wav')
    sf.write(temp_wav, data, sr)
    
    # Convert WAV to MP3
    audio = AudioSegment.from_wav(temp_wav)
    audio.export(output_path, format='mp3', bitrate='192k')
    
    # Remove temporary WAV file
    os.remove(temp_wav)


def process_audio(
        input_path, 
        pitch_increase=None,
        pitch_decrease=None,
        speed_increase=None,
        speed_decrease=None,
        reverb_room_size=None):
    """
    Process audio file with various transformations.
    
    Args:
        input_path: path to input audio file
        pitch_increase: positive number of semitones to increase pitch
        pitch_decrease: negative number of semitones to decrease pitch
        speed_increase: speed factor > 1.0 to increase speed
        speed_decrease: speed factor < 1.0 to decrease speed
        reverb_room_size: reverb room size (0.0 to 1.0)
    """

    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return
    
    # Validate inputs
    assert pitch_increase is None or pitch_increase > 0, "pitch_increase must be positive"
    assert pitch_decrease is None or pitch_decrease < 0, "pitch_decrease must be negative"
    assert speed_increase is None or speed_increase > 1.0, "speed_increase must be > 1.0"
    assert speed_decrease is None or (0 < speed_decrease < 1.0), "speed_decrease must be between 0 and 1"
    assert reverb_room_size is None or (0 < reverb_room_size <= 1.0), "reverb_room_size must be between 0 and 1"

    # Create output directory structure
    # - making a subdirectory within OUTPUT_DIR with the same name as the input file
    # - placing all modified audio files in this subdirectory
    # - moving the input file to this subdirectory too
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_subdir = os.path.join(OUTPUT_DIR, base_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    # Copy input file to output directory
    print(f"Copying input file to output directory...")
    original_copy_path = os.path.join(output_subdir, os.path.basename(input_path))
    shutil.copy2(input_path, original_copy_path)
    print(f"\tCopied to: {original_copy_path}\n")

    # Load audio
    print(f"Loading audio file ('{input_path}')...")
    data, sr = librosa.load(input_path, sr=None)

    # Apply Pitch Increase
    if pitch_increase:
        print(f"\tGenerating pitch increase by {pitch_increase} semitones...")
        pitch_up = shift_pitch(data, sr, n_steps=pitch_increase)
        output_path = os.path.join(output_subdir, f"{base_name}_pitch_up.mp3")
        save_as_mp3(pitch_up, sr, output_path)
        print(f"\t\tSaved: {output_path}\n")
    
    # Apply Pitch Decrease
    if pitch_decrease:
        print(f"\tGenerating pitch decrease by {pitch_decrease} semitones...")
        pitch_down = shift_pitch(data, sr, n_steps=pitch_decrease)
        output_path = os.path.join(output_subdir, f"{base_name}_pitch_down.mp3")
        save_as_mp3(pitch_down, sr, output_path)
        print(f"\t\tSaved: {output_path}\n")
    
    # Increase Speed
    if speed_increase:
        print(f"\tGenerating increased speed by factor of {speed_increase}...")
        speed_up = stretch_time(data, rate=speed_increase)
        output_path = os.path.join(output_subdir, f"{base_name}_speed_up.mp3")
        save_as_mp3(speed_up, sr, output_path)
        print(f"\t\tSaved: {output_path}\n")

    # Decrease Speed
    if speed_decrease:
        print(f"\tGenerating decreased speed by factor of {speed_decrease}...")
        speed_down = stretch_time(data, rate=speed_decrease)
        output_path = os.path.join(output_subdir, f"{base_name}_speed_down.mp3")
        save_as_mp3(speed_down, sr, output_path)
        print(f"\t\tSaved: {output_path}\n")

    # Apply Reverb
    if reverb_room_size:
        print(f"\tGenerating reverb (room size: {reverb_room_size})...")
        reverb = apply_reverb(data, sr, room_size=reverb_room_size)
        output_path = os.path.join(output_subdir, f"{base_name}_reverb.mp3")
        save_as_mp3(reverb, sr, output_path)
        print(f"\t\tSaved: {output_path}\n")

    print(f"Done! All files saved to: {output_subdir}\n")


if __name__ == "__main__":
    init_output_dir()

    process_audio(
        "input.mp3",
        pitch_increase=4,
        pitch_decrease=-4,
        speed_increase=1.5,
        speed_decrease=0.5,
        reverb_room_size=0.5
    )