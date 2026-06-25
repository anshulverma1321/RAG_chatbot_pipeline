import os
import sys
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from faster_whisper import WhisperModel
from RAG.exceptions import SpeechRecognitionError

# Use a reliable HF mirror to avoid connection blocks/errors on certain networks
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def record_audio(filename: str, samplerate: int = 16000) -> bool:
    """
    Records audio from the microphone until the user presses Enter.
    Saves the recorded audio as a WAV file at `filename`.
    Raises SpeechRecognitionError on failure.
    """
    q = queue.Queue()
    
    def callback(indata, frames, time, status):
        if status:
            print(f"Recording status: {status}", file=sys.stderr)
        q.put(indata.copy())
        
    try:
        # Check if microphone is available
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
        except Exception as dev_err:
            raise SpeechRecognitionError("Speech could not be recognized. Please try again.") from dev_err

        if not input_devices:
            raise SpeechRecognitionError("Speech could not be recognized. Please try again.")
            
        # Record audio
        with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
            print("\n[Listening... Speak your question now. Press ENTER to stop recording]")
            input() # Wait for user to press enter
            
        # Retrieve all blocks from queue
        audio_data = []
        while not q.empty():
            audio_data.append(q.get())
            
        if not audio_data:
            raise SpeechRecognitionError("Speech could not be recognized. Please try again.")
            
        # Concatenate and save to WAV
        audio_concat = np.concatenate(audio_data, axis=0)
        sf.write(filename, audio_concat, samplerate)
        return True
    except Exception as e:
        from RAG.logger import log_error
        log_error("speech_to_text.py", "record_audio", type(e).__name__, str(e))
        if isinstance(e, SpeechRecognitionError):
            raise
        raise SpeechRecognitionError("Speech could not be recognized. Please try again.") from e

class SpeechTranscriber:
    _instance = None
    
    @classmethod
    def get_instance(cls, model_size: str = "tiny"):
        """Singleton instance to avoid reloading the model on every query."""
        if cls._instance is None:
            # Check environment settings or default local D drive folder
            model_path = os.environ.get("WHISPER_MODEL_PATH", "D:/RAG Chatbot/models/faster-whisper-tiny")
            model_path = os.path.abspath(model_path)
            
            if os.path.exists(model_path) and os.path.isdir(model_path):
                cls._instance = cls(model_path)
            else:
                cls._instance = cls(model_size)
        return cls._instance
        
    def __init__(self, model_path_or_size: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        display_name = os.path.basename(model_path_or_size) if os.path.isdir(model_path_or_size) else model_path_or_size
        print(f"Loading Whisper model '{display_name}' on {self.device.upper()} (compute_type={self.compute_type})...")
        
        try:
            self.model = WhisperModel(model_path_or_size, device=self.device, compute_type=self.compute_type)
            print("Whisper model loaded successfully.")
        except Exception as e:
            print(f"Failed to load Whisper model on {self.device.upper()}: {e}")
            if self.device == "cuda":
                print("Falling back to CPU...")
                try:
                    self.device = "cpu"
                    self.compute_type = "int8"
                    self.model = WhisperModel(model_path_or_size, device=self.device, compute_type=self.compute_type)
                except Exception as cpu_err:
                    from RAG.logger import log_error
                    log_error("speech_to_text.py", "SpeechTranscriber.__init__", type(cpu_err).__name__, str(cpu_err))
                    raise SpeechRecognitionError("Speech could not be recognized. Please try again.") from cpu_err
            else:
                from RAG.logger import log_error
                log_error("speech_to_text.py", "SpeechTranscriber.__init__", type(e).__name__, str(e))
                raise SpeechRecognitionError("Speech could not be recognized. Please try again.") from e
                
    def transcribe(self, filepath: str) -> str:
        """Transcribes the audio file and returns the text."""
        import time
        import wave
        from RAG.logger import log_whisper, log_performance, log_error

        start_time = time.time()
        if not os.path.exists(filepath):
            raise SpeechRecognitionError("Speech could not be recognized. Please try again.")
            
        try:
            # Measure audio duration in seconds
            duration = 0.0
            try:
                with wave.open(filepath, 'r') as f:
                    frames = f.getnframes()
                    rate = f.getframerate()
                    duration = frames / float(rate)
            except Exception:
                pass
                
            segments, info = self.model.transcribe(filepath, beam_size=5)
            text = "".join(segment.text for segment in segments).strip()
            
            if not text:
                raise SpeechRecognitionError("Speech could not be recognized. Please try again.")
                
            elapsed = time.time() - start_time
            log_whisper(duration, text, "Success")
            log_performance({"whisper_transcription": elapsed})
            return text
        except Exception as e:
            log_error("speech_to_text.py", "transcribe", type(e).__name__, str(e))
            log_whisper(0.0, "", f"Failed: {e}")
            if isinstance(e, SpeechRecognitionError):
                raise
            raise SpeechRecognitionError("Speech could not be recognized. Please try again.") from e
