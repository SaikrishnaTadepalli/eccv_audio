import librosa
import soundfile as sf
import numpy as np
from scipy.signal import fftconvolve
from pydub import AudioSegment
import os
import shutil
from pathlib import Path
from typing import Optional

# MODIFY OUTPUT DIRECTORY HERE
OUTPUT_DIR = Path("./outputs/")

class AudioPermutationPipeline:
    """Pipeline for applying audio transformations and effects."""

    def __init__(self, output_dir: str = "./outputs/"):
        self.output_dir = Path(output_dir)
        self._init_output_dir()


    def _init_output_dir(self):
        """Ensures the output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


    @staticmethod
    def _shift_pitch(data, sr, n_steps=0):
        """
        Shifts pitch (without changing duration).

        Args:
            - positive 'n_steps' increases pitch
            - negative 'n_steps' decreases pitch
        """
        return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)


    @staticmethod
    def _stretch_time(data, rate=1.0):
        """
        Changes speed (without changing pitch).

        Args:
            - 'rate' > 1.0 increases speed
            - 'rate' < 1.0 decreases speed
        """
        return librosa.effects.time_stretch(y=data, rate=rate)


    @staticmethod
    def _apply_reverb(data, sr, room_size=0.5, wet_dry=0.3):
        """
        Adds reverb effect to audio.

        Args:
            - 'room_size' controls the reverb length (0.0 to 1.0)
            - 'wet_dry' mixes between original and reverb (0.0 = dry, 1.0 = wet)
        """
        reverb_duration = room_size * 2.0
        ir_length = int(reverb_duration * sr)

        t = np.linspace(0, reverb_duration, ir_length)
        decay = np.exp(-3.0 * t / reverb_duration)
        impulse = decay * np.random.randn(ir_length) * 0.1

        reverb_signal = fftconvolve(data, impulse, mode='same')
        output = (1 - wet_dry) * data + wet_dry * reverb_signal
        output = output / np.max(np.abs(output))

        return output


    @staticmethod
    def _overlay_audio(data, sr, overlay_path, volume_ratio=1.0):
        """
        Overlays a secondary audio file onto the original audio.
        
        Args:
            - data: original audio data (numpy array)
            - sr: sample rate of original audio
            - overlay_path: path to the secondary audio file to overlay
            - volume_ratio: volume of overlay relative to original (1.0 = equal volume, 0.5 = half volume)
        
        Returns:
            numpy array with overlaid audio
        """
        # Load the overlay audio file
        overlay_data, overlay_sr = librosa.load(overlay_path, sr=sr)
        
        # Get lengths
        original_length = len(data)
        overlay_length = len(overlay_data)
        
        # Handle length differences
        if overlay_length < original_length:
            # Loop the overlay to match original length
            num_repeats = int(np.ceil(original_length / overlay_length))
            overlay_data = np.tile(overlay_data, num_repeats)[:original_length]
        elif overlay_length > original_length:
            # Trim the overlay to match original length
            overlay_data = overlay_data[:original_length]
        
        # Apply volume ratio to overlay
        overlay_data = overlay_data * volume_ratio
        
        # Mix the two audio signals
        mixed = data + overlay_data
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0: mixed = mixed / max_val
        
        return mixed


    @staticmethod
    def _save_as_mp3(data, sr, output_path):
        """Save audio data as MP3 file."""
        temp_wav = str(output_path).replace('.mp3', '_temp.wav')
        sf.write(temp_wav, data, sr)

        audio = AudioSegment.from_wav(temp_wav)
        audio.export(output_path, format='mp3', bitrate='192k')

        os.remove(temp_wav)


    def process(self,
                input_path: str,
                pitch_increase: Optional[float] = None,
                pitch_decrease: Optional[float] = None,
                speed_increase: Optional[float] = None,
                speed_decrease: Optional[float] = None,
                reverb_room_size: Optional[float] = None):
        """
        Process audio file with various transformations.
        
        Args:
            - input_path: path to input audio file
            - pitch_increase: positive number of semitones to increase pitch
            - pitch_decrease: negative number of semitones to decrease pitch
            - speed_increase: speed factor > 1.0 to increase speed
            - speed_decrease: speed factor < 1.0 to decrease speed
            - reverb_room_size: reverb room size (0.0 to 1.0)
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

        # Create output subdirectory
        base_name = Path(input_path).stem
        output_subdir = self.output_dir / base_name
        output_subdir.mkdir(parents=True, exist_ok=True)

        # Copy input file
        print(f"Copying input file to output directory...")
        original_copy = output_subdir / Path(input_path).name
        shutil.copy2(input_path, original_copy)
        print(f"\tCopied to: {original_copy}\n")

        # Load audio
        print(f"Loading audio file ('{input_path}')...")
        data, sr = librosa.load(input_path, sr=None)

        # Apply transformations
        if pitch_increase:
            print(f"\tGenerating pitch increase by {pitch_increase} semitones...")
            pitch_up = self._shift_pitch(data, sr, n_steps=pitch_increase)
            output_path = output_subdir / f"{base_name}_pitch_up.mp3"
            self._save_as_mp3(pitch_up, sr, output_path)
            print(f"\t\tSaved: {output_path}\n")

        if pitch_decrease:
            print(f"\tGenerating pitch decrease by {pitch_decrease} semitones...")
            pitch_down = self._shift_pitch(data, sr, n_steps=pitch_decrease)
            output_path = output_subdir / f"{base_name}_pitch_down.mp3"
            self._save_as_mp3(pitch_down, sr, output_path)
            print(f"\t\tSaved: {output_path}\n")

        if speed_increase:
            print(f"\tGenerating increased speed by factor of {speed_increase}...")
            speed_up = self._stretch_time(data, rate=speed_increase)
            output_path = output_subdir / f"{base_name}_speed_up.mp3"
            self._save_as_mp3(speed_up, sr, output_path)
            print(f"\t\tSaved: {output_path}\n")

        if speed_decrease:
            print(f"\tGenerating decreased speed by factor of {speed_decrease}...")
            speed_down = self._stretch_time(data, rate=speed_decrease)
            output_path = output_subdir / f"{base_name}_speed_down.mp3"
            self._save_as_mp3(speed_down, sr, output_path)
            print(f"\t\tSaved: {output_path}\n")

        if reverb_room_size:
            print(f"\tGenerating reverb (room size: {reverb_room_size})...")
            reverb = self._apply_reverb(data, sr, room_size=reverb_room_size)
            output_path = output_subdir / f"{base_name}_reverb.mp3"
            self._save_as_mp3(reverb, sr, output_path)
            print(f"\t\tSaved: {output_path}\n")

        print(f"Done! All files saved to: {output_subdir}\n")


    def apply_overlay(self,
                      original_audio_path: str,
                      overlay_audio_path: str,
                      overlay_volume: float):
        """
        Apply audio overlay effect to an audio file.

        Args:
            - original_audio_path: path to the original audio file
            - overlay_audio_path: path to the overlay audio file
            - overlay_volume: volume of overlay relative to original (1.0 = equal, 0.5 = half)
        """
        # Check if original file exists
        if not os.path.exists(original_audio_path):
            print(f"Error: File '{original_audio_path}' not found.")
            return
        
        # Check if overlay file exists
        if not os.path.exists(overlay_audio_path):
            print(f"Error: Overlay file '{overlay_audio_path}' not found.")
            return
        
        # Create output subdirectory
        base_name = Path(original_audio_path).stem
        output_subdir = self.output_dir / base_name
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Copy input file
        print(f"Copying input file to output directory...")
        original_copy = output_subdir / Path(original_audio_path).name
        shutil.copy2(original_audio_path, original_copy)
        print(f"\tCopied to: {original_copy}\n")
        
        # Load audio
        print(f"Loading audio file ('{original_audio_path}')...")
        data, sr = librosa.load(original_audio_path, sr=None)
        
        # Apply overlay
        print(f"\tApplying overlay from '{overlay_audio_path}' at relative volume {overlay_volume}...")
        overlaid = self._overlay_audio(data, sr, overlay_audio_path, volume_ratio=overlay_volume)
        
        # Save output
        overlay_base_name = Path(overlay_audio_path).stem
        output_path = output_subdir / f"{base_name}_overlay_{overlay_base_name}.mp3"
        self._save_as_mp3(overlaid, sr, output_path)
        print(f"\t\tSaved: {output_path}\n")
        
        print(f"Done! File saved to: {output_subdir}\n")



if __name__ == "__main__":
    audio_pipeline = AudioPermutationPipeline(output_dir="./outputs/")

    audio_pipeline.process(
        "input.mp3",
        pitch_increase=4,
        pitch_decrease=-4,
        speed_increase=1.5,
        speed_decrease=0.5,
        reverb_room_size=0.5
    )

    background_noise_effects = [
        # https://www.youtube.com/watch?v=5jlUVr6gkos
        ("../saved_effects/wind.mp3", 0.75),

        # https://www.youtube.com/watch?v=C4pJ6Hi4MU4
        ("../saved_effects/rain.mp3", 1.25),

        # https://www.youtube.com/watch?v=wyzgbdI6x24
        ("../saved_effects/coffee_shop.mp3", 1.4),

        # https://www.youtube.com/watch?v=FeOrG8FrNko
        ("../saved_effects/busy_street.mp3", 0.6),

        # https://www.youtube.com/watch?v=cNWxqMx69WI
        ("../saved_effects/song1.mp3", 0.35),
    ]

    for background_audio_path, volume in background_noise_effects:
        audio_pipeline.apply_overlay(
            "input.mp3",
            background_audio_path,
            volume
        )
    