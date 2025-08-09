from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


def now_iso() -> str:
    """Return a timezone-aware ISO 8601 UTC timestamp.

    Consistent with JobStore timestamps (always UTC with 'Z' suffix for clarity).
    """
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace('+00:00', 'Z')

@dataclass
class JobRecord:
    id: str
    user_id: str
    status: JobStatus
    file_name: str
    input_blob_url: str
    attenuation_db: int = 30
    processing_type: str = "denoise"
    progress: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        base = dict(
            id=self.id,
            userId=self.user_id,
            status=self.status.value,
            fileName=self.file_name,
            input_blob_url=self.input_blob_url,
            attenuationDb=self.attenuation_db,
            processingType=self.processing_type,
            progress=self.progress,
            **self.metadata,
        )
        return base

    @classmethod
    def from_cosmos(cls, item: Dict[str, Any]) -> "JobRecord":
        raw_att = item.get('attenuationDb', 30)
        try:
            att = int(raw_att)
        except (ValueError, TypeError):
            att = 30
        # Clamp to defensible bounds (model heuristic range)
        if att < -10:
            att = -10
        elif att > 80:
            att = 80

        return cls(
            id=item.get('id') or item.get('jobId'),
            user_id=item.get('userId') or item.get('user_id'),
            status=JobStatus(item.get('status', 'queued')),
            file_name=item.get('fileName') or item.get('filename') or 'unknown',
            input_blob_url=item.get('input_blob_url') or item.get('inputBlobUrl'),
            attenuation_db=att,
            processing_type=item.get('processingType', 'denoise'),
            progress=item.get('progress', 0),
            metadata={k: v for k, v in item.items() if k not in {'id','userId','status','fileName','input_blob_url','attenuationDb','processingType','progress'}},
        )
