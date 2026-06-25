import os
import sys
import subprocess
import threading
import tempfile
import re
import sounddevice as sd
import soundfile as sf
from RAG.exceptions import TextToSpeechError

class TextToSpeechEngine:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Singleton instance to avoid reloading engine metadata on every query."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        # Retrieve settings from environment or default to local workspace paths
        default_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_piper_exe = os.path.join(default_dir, "models", "piper", "piper.exe")
        default_piper_model = os.path.join(default_dir, "models", "piper", "en_US-lessac-medium.onnx")

        self.piper_exe = os.environ.get("PIPER_EXE", default_piper_exe)
        self.piper_model = os.environ.get("PIPER_MODEL", default_piper_model)
        
        # Normalize paths
        self.piper_exe = os.path.abspath(self.piper_exe)
        self.piper_model = os.path.abspath(self.piper_model)
        
        self.available = True
        missing_resources = []
        if not os.path.exists(self.piper_exe):
            print(f"[TTS Warning] Piper executable not found at: {self.piper_exe}", file=sys.stderr)
            self.available = False
            missing_resources.append("piper.exe")
            
        if not os.path.exists(self.piper_model):
            print(f"[TTS Warning] Piper model ONNX not found at: {self.piper_model}", file=sys.stderr)
            self.available = False
            missing_resources.append("voice model")
            
        if not self.available:
            from RAG.logger import log_error
            # Log the TextToSpeechError for developer diagnostics
            try:
                raise TextToSpeechError(
                    f"Voice playback unavailable (missing: {', '.join(missing_resources)}). "
                    f"Text response will still be shown."
                )
            except TextToSpeechError as e:
                log_error("text_to_speech.py", "__init__", type(e).__name__, str(e))
        
        self.play_queue = []
        self.queue_lock = threading.Lock()
        self.last_text = ""
        self.voice_enabled = os.environ.get("VOICE_ENABLED", "true").lower() == "true"
        
        if self.available:
            self._start_player_thread()
            
    def _start_player_thread(self):
        self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.play_thread.start()
        
    def _play_loop(self):
        import time
        from RAG.logger import log_error
        while True:
            filepath = None
            with self.queue_lock:
                if self.play_queue:
                    filepath = self.play_queue.pop(0)
            
            if filepath:
                try:
                    data, fs = sf.read(filepath)
                    sd.play(data, fs)
                    sd.wait()
                except Exception as e:
                    # Log playback error to errors.log and print user-friendly warning
                    try:
                        raise TextToSpeechError("Audio playback failed.") from e
                    except TextToSpeechError as tts_err:
                        log_error("text_to_speech.py", "_play_loop", type(tts_err).__name__, str(tts_err))
                    print(f"\n[TTS Warning] Voice playback unavailable. Text response will still be shown.", file=sys.stderr)
                finally:
                    if os.path.exists(filepath):
                        try:
                            os.remove(filepath)
                        except Exception:
                            pass
            else:
                time.sleep(0.1)
                
    def speak(self, text: str, wait_for_finish: bool = False):
        if not self.available or not self.voice_enabled:
            return
            
        # Clean text (remove citations and markdown formatting)
        cleaned_text = self._clean_citations(text)
        self.last_text = text # Store original response text for repeat/replay
        
        # Stop any currently playing speech immediately
        sd.stop()
        with self.queue_lock:
            # Clear old queued wavs and delete their files
            for old_file in self.play_queue:
                if os.path.exists(old_file):
                    try:
                        os.remove(old_file)
                    except Exception:
                        pass
            self.play_queue.clear()
            
        # Write to temporary file path
        temp_dir = tempfile.gettempdir()
        temp_wav = os.path.join(temp_dir, f"piper_tts_{threading.get_ident()}_{os.getpid()}.wav")
        
        def run_tts():
            import time
            from RAG.logger import log_piper, log_performance, log_error
            start_time = time.time()
            try:
                import wave
                from piper import PiperVoice
                
                # Lazy load voice model on first synthesis
                if not hasattr(self, 'voice') or self.voice is None:
                    self.voice = PiperVoice.load(self.piper_model)
                
                # Generate voice WAV file directly
                with wave.open(temp_wav, "wb") as wav_file:
                    self.voice.synthesize_wav(cleaned_text, wav_file)
                
                # Check output file
                if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
                    elapsed = time.time() - start_time
                    voice_name = os.path.basename(self.piper_model).replace(".onnx", "")
                    log_piper(len(cleaned_text), voice_name, os.path.basename(temp_wav))
                    log_performance({"piper_generation": elapsed})
                    with self.queue_lock:
                        self.play_queue.append(temp_wav)
                else:
                    raise TextToSpeechError("Piper output WAV file is missing or empty.")
            except Exception as e:
                # Log synthesis error to errors.log and print user-friendly warning
                try:
                    if isinstance(e, TextToSpeechError):
                        raise
                    raise TextToSpeechError("Piper audio generation failed.") from e
                except TextToSpeechError as tts_err:
                    log_error("text_to_speech.py", "speak.run_tts", type(tts_err).__name__, str(tts_err))
                print(f"\n[TTS Warning] Voice playback unavailable. Text response will still be shown.", file=sys.stderr)
                
        if wait_for_finish:
            run_tts()
            # Wait for file to clear from queue (meaning it finished playing)
            import time
            while True:
                with self.queue_lock:
                    in_queue = temp_wav in self.play_queue
                if not in_queue:
                    break
                time.sleep(0.05)
        else:
            # Run asynchronously in background thread
            threading.Thread(target=run_tts, daemon=True).start()
            
    def repeat(self):
        if self.last_text:
            self.speak(self.last_text)
            
    def stop(self):
        sd.stop()
        with self.queue_lock:
            for filepath in self.play_queue:
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception:
                        pass
            self.play_queue.clear()
            
    def _clean_citations(self, text: str) -> str:
        """Cleans citations and markdown tokens from text so it reads naturally."""
        # Remove markdown citation blocks like [documentation.pdf (Page 11)]
        cleaned = re.sub(r'\[[^\]]+\.pdf\s+\(Page[s]?\s+\d+(?:-\d+)?\)\]', '', text)
        # Remove formatting symbols
        cleaned = cleaned.replace("*", "")
        cleaned = cleaned.replace("`", "")
        cleaned = cleaned.replace("", "") # clean any cp1252 parsing replacement characters
        return cleaned.strip()
