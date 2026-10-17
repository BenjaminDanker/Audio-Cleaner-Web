"""What a live session does: noise removal, captions, the spoken languages, translations, echo removal and the noise
strength (docs/streaming-protocol.md). The client's `start` message sets all of it and a `config` message changes any
part of it while the session runs, under the same rules (parse). What it costs per minute follows from it (rate)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .. import translate
from ..billing import pricing
from ..config import get_settings


class FeatureError(Exception):
    """A start or config message asks for something that can't be done: `code` and a message for the person."""

    def __init__(self, code: str, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.code, self.message, self.retry_after = code, message, retry_after

    def event(self) -> dict:
        return {"type": "error", "code": self.code, "message": self.message,
                **({"retryAfterSeconds": self.retry_after} if self.retry_after else {})}


@dataclass(frozen=True)
class Features:
    denoise: str | None = None  # "light"
    captions: str | None = None  # "local" (Standard) or "premium" (extra accurate)
    languages: tuple[str, ...] = ()  # spoken: none (detect), one, or up to three for people who switch between them
    translate: tuple[str, ...] = ()  # caption translations
    echo_reference: bool = False
    atten_lim_db: float = 100.0

    @property
    def language(self) -> str | None:
        """The spoken language when it is one for sure."""
        return self.languages[0] if len(self.languages) == 1 else None

    @property
    def caption_language(self) -> str | None:
        """The language of the captions, as far as it is known (what translations translate from)."""
        return self.language or ("en" if self.captions == "local" and not self.languages else None)

    def rate(self) -> int:
        """Micros per minute of audio (translations are billed by the speech translated, separately)."""
        total = 0
        if self.denoise:
            total += pricing.rate("stream_denoise", self.denoise).micros_per_minute
        if self.captions:
            total += pricing.rate("stream_captions", self.captions).micros_per_minute
        return total

    def premium_rate(self) -> int:
        """The extra-accurate captions' part of the rate (0 without them)."""
        return pricing.rate("stream_captions", "premium").micros_per_minute if self.captions == "premium" else 0

    def trial_eligible(self) -> bool:
        """The free trial covers Standard live only: noise removal and/or Standard captions."""
        return self.denoise in (None, "light") and self.captions in (None, "local") and not self.translate

    def worker(self, trial: bool) -> dict:
        """What the live engine is told (start and MSG_CONFIG): it runs noise removal, Standard captions and echo
        removal; for extra-accurate captions it only hands over the audio (the tap)."""
        return {"denoise": self.denoise, "captions": "local" if self.captions == "local" else None,
                "echo_reference": self.echo_reference, "tap_pcm": self.captions == "premium", "trial": trial,
                "atten_lim_db": self.atten_lim_db}


def atten(value) -> float | None:
    """Noise-strength limit in dB, 6 to 100; None when it isn't a finite number."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(v, 6.0), 100.0) if math.isfinite(v) else None


def parse(msg: dict, current: Features | None = None) -> Features:
    """The features of a `start` message (`current` None: what it leaves out is off), or `current` changed by a
    `config` message (what it leaves out stays). Raises FeatureError."""
    base = current or Features()
    denoise = msg["denoise"] if "denoise" in msg else base.denoise
    captions = msg["captions"] if "captions" in msg else base.captions
    if not all(v is None or isinstance(v, str) for v in (denoise, captions)):
        raise FeatureError("bad_config", "Check the noise strength and the options you chose.")
    languages = base.languages
    if "language" in msg:
        spoken = msg["language"]
        raw = spoken if isinstance(spoken, list) else ([] if spoken is None else [spoken])
        if not all(isinstance(x, str) and 2 <= len(x) <= 3 and x.isalpha() for x in raw) or len(raw) > 3:
            raise FeatureError("bad_config", "A 2-3 letter language code, or a list of up to three.")
        languages = tuple(dict.fromkeys(x.lower() for x in raw))
    echo = msg["echoReference"] is True if "echoReference" in msg else base.echo_reference
    strength = base.atten_lim_db
    if "attenLimDb" in msg:
        strength = atten(msg["attenLimDb"])
        if strength is None:
            raise FeatureError("bad_config", "Check the noise strength and the options you chose.")
    f = Features(denoise, captions, languages, base.translate, echo, strength)
    try:
        rate = f.rate()
    except pricing.PricingError as e:
        raise FeatureError("bad_tier", str(e)) from None
    if rate == 0:
        raise FeatureError("nothing_enabled", "Enable denoise and/or captions.")
    if captions == "local" and languages not in ((), ("en",)):
        raise FeatureError("bad_config", "Standard captions are English only. Extra-accurate captions work in other languages.")
    # Translations are checked against the caption language they would translate from, also when only that changed.
    wanted = msg["translate"] if "translate" in msg else list(base.translate)
    try:
        targets = tuple(translate.check_targets(wanted, f.caption_language))
    except ValueError as e:
        raise FeatureError("bad_config", f"Translations: {e}.") from None
    if targets and not captions:
        raise FeatureError("bad_config", "Translations need captions: turn captions on too.")
    openai = bool(get_settings().openai_api_key)
    if targets and not openai:
        raise FeatureError("unavailable", "Translations aren’t available right now. Try without them.", 300)
    if captions == "premium" and not openai:
        raise FeatureError("unavailable", "Extra-accurate captions aren’t available right now.", 300)
    return Features(denoise, captions, languages, targets, echo, strength)
