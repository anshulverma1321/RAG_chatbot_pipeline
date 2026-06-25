import os
import sys

# Ensure root folder is in sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(base_dir, ".env"))

import wave
from piper import PiperVoice
import sounddevice as sd
import soundfile as sf

def safe_print(text):
    """Prints text safely, replacing unencodable characters on Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(text.encode(encoding, errors='replace').decode(encoding))

def main():
    text = "This is a demonstration of the Piper text-to-speech engine."
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        
    out_dir = os.path.join(base_dir, "outputs", "piper")
    os.makedirs(out_dir, exist_ok=True)
    out_wav = os.path.join(out_dir, "synthesis.wav")
    
    # Piper Model configuration
    default_model = os.path.join(base_dir, "models", "piper", "en_US-lessac-medium.onnx")
    piper_model = os.environ.get("PIPER_MODEL", default_model)
    
    if not os.path.exists(piper_model):
        print(f"Error: Piper voice model not found at: {piper_model}")
        sys.exit(1)
        
    print("Generating Audio...")
    try:
        # 1. Load voice model
        voice = PiperVoice.load(piper_model)
        
        # 2. Synthesize to output WAV
        with wave.open(out_wav, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
            
        print(f"Audio Saved:")
        print(os.path.basename(out_wav))
        
        # 3. Play audio automatically using sounddevice/soundfile
        print("\nPlaying audio...")
        data, fs = sf.read(out_wav)
        sd.play(data, fs)
        sd.wait()
        print("Playback complete.")
        
    except Exception as e:
        print(f"Error generating or playing Piper audio: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
