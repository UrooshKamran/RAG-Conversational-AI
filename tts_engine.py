"""
tts_engine.py
Ensures audio is yielded in distinct chunks for the WebSocket.
"""
import io
import wave
from piper.voice import PiperVoice

class TTSEngine:
    def __init__(self, model_path: str = "voices/en_US-lessac-medium.onnx"):
        self.voice = PiperVoice.load(model_path)
        self.sample_rate = self.voice.config.sample_rate

    def _pcm_to_wav(self, pcm_bytes: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm_bytes)
        return buf.getvalue()

    def synthesize_streaming(self, text: str):
        """
        Yields a proper WAV byte-string for every sentence detected by Piper.
        """
        if not text.strip():
            return

        # Piper's synthesize generator yields AudioChunks per sentence/phrase
        for chunk in self.voice.synthesize(text):
            pcm_bytes = chunk.audio_int16_bytes
            if pcm_bytes:
                # Wrap each chunk in its own WAV header so the browser can decode it
                yield self._pcm_to_wav(pcm_bytes)