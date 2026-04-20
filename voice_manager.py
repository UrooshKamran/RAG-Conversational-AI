"""
voice_manager.py
Parallel display + speech: tokens stream to screen sentence by sentence,
and each sentence is synthesized and sent as audio immediately when complete.
This gives natural word-by-word display with synchronized speech.
"""
import re
from asr_engine import ASREngine
from tts_engine import TTSEngine
from conversation_manager import ConversationManager

# Sentence boundary pattern
SENTENCE_END = re.compile(r'([.!?,;:])\s+|([.!?])\s*$')

def _split_into_sentences(text: str) -> list:
    """Split text into sentences for TTS chunking."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


class VoiceManager:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.asr  = ASREngine(model_size="tiny.en")
        self.tts  = TTSEngine(model_path="voices/en_US-lessac-medium.onnx")
        self.conv = ConversationManager(session_id=session_id)

    def process_audio_streaming(self, audio_bytes: bytes):
        """
        Pipeline: ASR -> LLM stream -> parallel token display + sentence TTS.

        For each sentence completed during LLM streaming:
          1. Yield all its tokens (display them word by word)
          2. Immediately synthesize and yield the audio for that sentence
        This makes display and speech happen in sync, sentence by sentence.
        """
        # 1. Transcribe
        user_text = self.asr.transcribe_bytes(audio_bytes)
        if not user_text:
            yield {"type": "error", "data": "Could not understand audio."}
            return

        yield {"type": "transcript", "data": user_text}

        # 2. Stream LLM tokens, buffer into sentences, speak each sentence
        sentence_buffer = ""
        token_buffer    = []   # tokens belonging to current sentence
        full_response   = ""

        for token in self.conv.stream_chat(user_text):
            full_response   += token
            sentence_buffer += token
            token_buffer.append(token)

            # Check if we've completed a sentence
            if re.search(r'[.!?]["\']?\s*$', sentence_buffer.rstrip()):
                # Yield all tokens for this sentence first (display)
                for t in token_buffer:
                    yield {"type": "token", "data": t}

                # Then immediately synthesize and yield audio for this sentence
                sentence_text = sentence_buffer.strip()
                if sentence_text:
                    for audio_chunk in self.tts.synthesize_streaming(sentence_text):
                        yield {"type": "audio", "data": audio_chunk}

                # Reset buffers
                sentence_buffer = ""
                token_buffer    = []

        # Handle any remaining text that didn't end with punctuation
        if token_buffer:
            for t in token_buffer:
                yield {"type": "token", "data": t}
            if sentence_buffer.strip():
                for audio_chunk in self.tts.synthesize_streaming(sentence_buffer.strip()):
                    yield {"type": "audio", "data": audio_chunk}

        # 3. Done
        yield {
            "type": "done",
            "cart": self.conv.cart.get_summary(),
            "session_active": self.conv.is_active
        }

    def reset(self):
        self.conv.reset_session()

    def get_state(self) -> dict:
        return self.conv.get_session_state()
