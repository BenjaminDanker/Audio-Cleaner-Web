"""ASR + cleanup + translation pipeline using Azure AI Speech Services and OpenAI for cleanup.

Environment variables expected:
  - AZURE_SPEECH_SERVICES_KEY
  - AZURE_SPEECH_SERVICES_REGION
  - AZURE_OPENAI_ENDPOINT
  - AZURE_OPENAI_API_KEY
  - AZURE_OPENAI_API_VERSION (e.g., 2024-06-01)
  - AZURE_OPENAI_CLEANUP_DEPLOYMENT (small chat model for cleanup, e.g., gpt-4.1-nano)
  - AZURE_TRANSLATOR_KEY
  - AZURE_TRANSLATOR_ENDPOINT (e.g., https://api.cognitive.microsofttranslator.com)
  - AZURE_TRANSLATOR_REGION

Outputs:
  - File mode returns SubtitleBundle with per-language timed segments
  - Streaming mode returns SubtitleDelta for the latest chunk
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
import io
import json
import requests
import tempfile

import soundfile as sf

try:
    from openai import AzureOpenAI  # type: ignore
except Exception:  # pragma: no cover - dependency optional in dev
    AzureOpenAI = None  # type: ignore

try:
    import azure.cognitiveservices.speech as speechsdk  # type: ignore
except Exception:  # pragma: no cover - dependency optional in dev
    speechsdk = None  # type: ignore


@dataclass
class SubtitleSegment:
    start: float
    end: float
    text: str


@dataclass
class SubtitleBundle:
    # lang_code -> list of segments
    segments_by_lang: Dict[str, List[SubtitleSegment]] = field(default_factory=dict)
    # Primary language code (e.g., 'en')
    primary_lang: str = "en"


@dataclass
class SubtitleDelta:
    # For streaming, only latest segments per language
    segments_by_lang: Dict[str, List[SubtitleSegment]] = field(default_factory=dict)
    is_final: bool = False


SYSTEM_CLEAN_PROMPT = (
    "You are a caption cleaner. Fix punctuation, casing, spacing, obvious ASR artifacts, and numeral formatting. "
    "DO NOT add or remove facts. DO NOT change the meaning. Preserve timings; only adjust text content."
)


def _get_openai_client():
    if AzureOpenAI is None:
        raise RuntimeError("openai package not installed; add 'openai' to requirements")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    if not endpoint or not api_key:
        raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY")
    client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
    return client


def _get_speech_services_config():
    """Get Azure Speech Services configuration."""
    if speechsdk is None:
        raise RuntimeError("azure-cognitiveservices-speech package not installed")
    
    subscription_key = os.getenv("AZURE_SPEECH_SERVICES_KEY")
    region = os.getenv("AZURE_SPEECH_SERVICES_REGION")
    
    if not subscription_key or not region:
        raise RuntimeError("Missing AZURE_SPEECH_SERVICES_KEY or AZURE_SPEECH_SERVICES_REGION")
    
    return speechsdk.SpeechConfig(subscription=subscription_key, region=region)


def _transcribe_file_with_speech_services(wav_path: str, translate_to_english: bool = False) -> Tuple[List[SubtitleSegment], str]:
    """Call Azure AI Speech Services for batch transcription. Returns (segments, detected_lang)."""
    speech_config = _get_speech_services_config()
    
    # Configure for batch transcription with detailed output
    speech_config.output_format = speechsdk.OutputFormat.Detailed
    speech_config.request_word_level_timestamps()
    
    # Audio configuration
    audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
    
    # Create speech recognizer
    if translate_to_english:
        # Use speech translation if needed
        translation_config = speechsdk.translation.SpeechTranslationConfig(
            subscription=speech_config.subscription_key, 
            region=speech_config.region
        )
        translation_config.add_target_language("en")
        recognizer = speechsdk.translation.TranslationRecognizer(
            translation_config=translation_config, 
            audio_config=audio_config
        )
    else:
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    
    # Perform recognition
    result = recognizer.recognize_once()
    
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        # Parse the detailed result to extract segments with timestamps
        segments = []
        if hasattr(result, 'json') and result.json:
            json_result = json.loads(result.json)
            if 'NBest' in json_result and json_result['NBest']:
                words = json_result['NBest'][0].get('Words', [])
                if words:
                    # Group words into segments (roughly every 5-10 seconds)
                    current_text = ""
                    start_time = 0
                    last_end_time = 0
                    
                    for i, word in enumerate(words):
                        if not current_text:
                            start_time = word.get('Offset', 0) / 10000000  # Convert to seconds
                        
                        current_text += word.get('Word', '') + " "
                        last_end_time = (word.get('Offset', 0) + word.get('Duration', 0)) / 10000000
                        
                        # Create segment every ~5 seconds or at end
                        if (last_end_time - start_time >= 5.0) or (i == len(words) - 1):
                            segments.append(SubtitleSegment(
                                start=start_time,
                                end=last_end_time,
                                text=current_text.strip()
                            ))
                            current_text = ""
                else:
                    # Fallback: single segment with full text
                    segments.append(SubtitleSegment(
                        start=0.0,
                        end=result.duration.total_seconds() if hasattr(result, 'duration') else 10.0,
                        text=result.text
                    ))
        else:
            # Fallback: single segment
            segments.append(SubtitleSegment(
                start=0.0,
                end=10.0,  # Default duration
                text=result.text
            ))
        
        # Detect language (Speech Services doesn't return this directly, assume English)
        detected_language = "en"
        return segments, detected_language
        
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return [], "en"
    else:
        error_details = result.error_details if hasattr(result, 'error_details') else "Unknown error"
        raise RuntimeError(f"Speech recognition failed: {error_details}")


def _cleanup_segments_with_llm(segments: List[SubtitleSegment]) -> List[SubtitleSegment]:
    if not segments:
        return segments
    client = _get_openai_client()
    chat_depl = os.getenv("AZURE_OPENAI_CLEANUP_DEPLOYMENT", "gpt-4.1-nano")
    # Prepare a compact JSON payload for the model
    payload = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    content = json.dumps(payload, ensure_ascii=False)
    chat = client.chat.completions.create(
        model=chat_depl,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_CLEAN_PROMPT},
            {"role": "user", "content": f"Clean these segments. Return JSON array with same start/end fields and cleaned 'text'.\n{content}"},
        ],
    )
    out_txt = chat.choices[0].message.content if chat.choices else "[]"
    try:
        cleaned = json.loads(out_txt)
    except Exception:
        # Fallback: pass-through
        return segments
    result: List[SubtitleSegment] = []
    for s in cleaned:
        result.append(SubtitleSegment(start=float(s.get("start", 0.0)), end=float(s.get("end", 0.0)), text=str(s.get("text", "")).strip()))
    return result


def _translate_segments(segments: List[SubtitleSegment], to_lang: str) -> List[SubtitleSegment]:
    if not segments:
        return segments
    key = os.getenv("AZURE_TRANSLATOR_KEY")
    endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
    region = os.getenv("AZURE_TRANSLATOR_REGION")
    if not key or not endpoint or not region:
        raise RuntimeError("Missing Azure Translator env (AZURE_TRANSLATOR_KEY/ENDPOINT/REGION)")
    path = "/translate?api-version=3.0"
    params = f"&to={to_lang}"
    url = endpoint.rstrip("/") + path + params
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Ocp-Apim-Subscription-Region": region,
        "Content-type": "application/json",
    }
    batch = [{"Text": s.text} for s in segments]
    resp = requests.post(url, headers=headers, json=batch, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    out: List[SubtitleSegment] = []
    for s, item in zip(segments, data):
        tr = item["translations"][0]["text"] if item.get("translations") else s.text
        out.append(SubtitleSegment(start=s.start, end=s.end, text=tr))
    return out


def transcribe_and_translate_file(wav_path: str, sr: int, target_langs: List[str]) -> SubtitleBundle:
    """Main transcription function using Azure AI Speech Services."""
    # Speech Services: if requested English translation, set translate flag
    translate_direct = False
    if len(target_langs) == 1 and target_langs[0].lower() == "en":
        translate_direct = True
    
    segs, detected_lang = _transcribe_file_with_speech_services(wav_path, translate_to_english=translate_direct)
    cleaned = _cleanup_segments_with_llm(segs)
    bundle = SubtitleBundle(primary_lang=("en" if translate_direct else detected_lang))
    bundle.segments_by_lang[bundle.primary_lang] = cleaned
    
    # Additional translations
    for lang in target_langs:
        lc = lang.lower()
        if lc == bundle.primary_lang:
            continue
        translated = _translate_segments(cleaned, to_lang=lc)
        bundle.segments_by_lang[lc] = translated
    return bundle


# Streaming mode: buffer management would be handled by caller; here we provide a chunk method

def transcribe_and_translate_chunk(processed_chunk_bytes: bytes, sr: int, target_langs: List[str], state: Optional[dict] = None) -> 'SubtitleDelta':
    # Build a small WAV in-memory for Speech Services processing
    bio = io.BytesIO()
    sf.write(bio, sf.read(io.BytesIO(processed_chunk_bytes))[0] if False else [], sr)  # placeholder to keep type hints
    # In practice, the caller should provide chunk PCM; here we assume the chunk has been persisted separately.
    # For now, we raise NotImplementedError to avoid misleading partial implementation.
    raise NotImplementedError("Streaming transcription is handled by the streaming service; use file mode here.")
