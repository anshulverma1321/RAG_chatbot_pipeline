import os
import sys

# Ensure root folder is in sys.path
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(base_dir, ".env"))

from RAG.speech_to_text import record_audio, SpeechTranscriber

def safe_print(text):
    """Prints text safely, replacing unencodable characters on Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(text.encode(encoding, errors='replace').decode(encoding))

def main():
    out_dir = os.path.join(base_dir, "outputs", "whisper")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Determine input source (mic or file)
    audio_path = None
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        print(f"Running Whisper transcription on file: {audio_path}")
    else:
        # No argument, record from microphone
        audio_path = os.path.join(out_dir, "recorded_test.wav")
        print("No audio file provided. Starting microphone recording...")
        try:
            success = record_audio(audio_path)
            if not success:
                print("Error: Audio recording failed.")
                sys.exit(1)
            print(f"Audio recorded and saved to: {audio_path}")
        except Exception as e:
            print(f"Error during audio recording: {e}")
            sys.exit(1)
            
    # 2. Run Whisper transcription
    try:
        print("Loading Whisper transcriber (lazy load)...")
        transcriber = SpeechTranscriber.get_instance(model_size="tiny")
        
        print("Transcribing audio...")
        text = transcriber.transcribe(audio_path)
        
        print("\nRecognized Text:")
        safe_print(text if text else "[Empty transcription]")
        
        # Save output
        out_file = os.path.join(out_dir, "transcription.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)
            
        print(f"\nTranscription saved to: {out_file}")
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
