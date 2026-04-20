"""
asr_engine.py
Automatic Speech Recognition using faster-whisper.
Captures audio from microphone and transcribes to text.
Uses whisper tiny.en model for low-latency CPU inference.
"""

import io
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel


class ASREngine:
    """
    Handles microphone audio capture and Whisper transcription.
    
    Strategy:
    - Record audio for a fixed duration (non-streaming, simple approach)
    - Transcribe using faster-whisper tiny.en model
    - Return transcribed text to the voice pipeline
    """

    SAMPLE_RATE    = 16000   # Whisper expects 16kHz
    CHANNELS       = 1       # Mono
    RECORD_SECONDS = 5       # Max recording duration per utterance
    SILENCE_THRESHOLD = 0.01 # RMS threshold to detect silence

    def __init__(self, model_size: str = "tiny.en"):
        """
        Load the Whisper model on initialization.
        tiny.en is ~40MB and runs in <1s on CPU.
        """
        print(f"[ASR] Loading Whisper model: {model_size}...")
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"   # int8 quantization for faster CPU inference
        )
        print("[ASR] Whisper model loaded.")

    def record_audio(self, duration: int = None) -> np.ndarray:
        """
        Record audio from the default microphone.
        Returns numpy array of audio samples.
        """
        duration = duration or self.RECORD_SECONDS
        print(f"[ASR] Recording for {duration}s...")
        audio = sd.rec(
            int(duration * self.SAMPLE_RATE),
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype="float32"
        )
        sd.wait()  # Wait until recording is done
        print("[ASR] Recording complete.")
        return audio.flatten()

    def record_until_silence(self, max_duration: int = 8, silence_duration: float = 1.0) -> np.ndarray:
        """
        Record audio and stop automatically when silence is detected.
        More natural than fixed-duration recording.
        """
        chunk_size = int(self.SAMPLE_RATE * 0.1)  # 100ms chunks
        max_chunks = int(max_duration / 0.1)
        silence_chunks = int(silence_duration / 0.1)

        audio_chunks = []
        silent_count = 0
        started_speaking = False

        print("[ASR] Listening... (speak now)")

        with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=self.CHANNELS, dtype="float32") as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_size)
                chunk = chunk.flatten()
                audio_chunks.append(chunk)

                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if rms > self.SILENCE_THRESHOLD:
                    started_speaking = True
                    silent_count = 0
                elif started_speaking:
                    silent_count += 1
                    if silent_count >= silence_chunks:
                        print("[ASR] Silence detected, stopping.")
                        break

        return np.concatenate(audio_chunks)

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe a numpy audio array to text using Whisper.
        Returns transcribed string.
        """
        # faster-whisper expects float32 numpy array at 16kHz
        segments, info = self.model.transcribe(
            audio,
            language="en",
            beam_size=1,          # beam_size=1 is fastest
            vad_filter=True,      # filter out silence automatically
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        text = " ".join(segment.text.strip() for segment in segments)
        print(f"[ASR] Transcribed: '{text}'")
        return text.strip()

    def record_and_transcribe(self, use_silence_detection: bool = True) -> str:
        """
        Convenience method: record audio then transcribe.
        Returns transcribed text string.
        """
        if use_silence_detection:
            audio = self.record_until_silence()
        else:
            audio = self.record_audio()
        return self.transcribe(audio)

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio from raw bytes (for WebSocket audio streaming).
        Handles webm/ogg format from browser MediaRecorder.
        """
        import subprocess
        import tempfile
        import os

        # Write incoming bytes to temp file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(audio_bytes)
            webm_path = f.name

        wav_path = webm_path.replace('.webm', '.wav')

        try:
            # Convert webm to wav using ffmpeg
            subprocess.run([
                'ffmpeg', '-y', '-i', webm_path,
                '-ar', '16000',   # 16kHz for Whisper
                '-ac', '1',       # mono
                '-f', 'wav',
                wav_path
            ], capture_output=True, check=True)

            # Read converted wav
            sample_rate, audio_data = wav.read(wav_path)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                if audio_data.max() > 1.0:
                    audio_data /= 32768.0

            return self.transcribe(audio_data)

        except subprocess.CalledProcessError as e:
            print(f"[ASR] ffmpeg error: {e.stderr.decode()}")
            return ""
        finally:
            os.unlink(webm_path)
            if os.path.exists(wav_path):
                os.unlink(wav_path)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asr = ASREngine(model_size="tiny.en")
    print("\nSpeak something and press Enter when done...")
    text = asr.record_and_transcribe(use_silence_detection=True)
    print(f"\nResult: '{text}'")