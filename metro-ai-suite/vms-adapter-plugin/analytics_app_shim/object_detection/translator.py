# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DLStreamer metadata → Nx analytics object push payload translator.

Converts the inference metadata published by DLStreamer Pipeline Server
to MQTT into the list of Nx object-push dicts expected by
``POST /rest/v4/analytics/engines/{engineId}/deviceAgents/{deviceId}/metadata/object``.

Sample DLS payload shape (see sample_app_metadata.json):
{
  "objects": [
    {
      "detection": {
        "bounding_box": {"x_min": 0.87, "x_max": 0.99, "y_min": 0.16, "y_max": 0.31},
        "confidence": 0.745,
        "label": "car",
        "label_id": 2
      },
      "region_id": 1,
      "roi_type": "car",
      ...
    }
  ],
  "rtp": {"sender_ntp_unix_timestamp_ns": 1777350580751188754},
  "timestamp": 66611537331
}
"""

from __future__ import annotations

import time
import uuid
from typing import Any


_TYPE_DEFAULT = "python.detected.object"
_MAX_TIMESTAMP_SKEW_MS = 750


def _ns_epoch_to_ms(value: Any) -> int:
    """Convert an epoch-nanoseconds value to epoch-milliseconds."""
    try:
        ns = int(value)
    except (TypeError, ValueError):
        return 0
    return ns // 1_000_000 if ns > 0 else 0


def _label_to_type_id(label: str, label_type_map: dict[str, str]) -> str:
    """Resolve a detection label to a registered Nx typeId.

    Looks up ``label`` (case-insensitively) in ``label_type_map``.
    Falls back to ``python.detected.object`` for unrecognised labels.
    """
    return label_type_map.get(label.lower(), _TYPE_DEFAULT)


def _to_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion for optional numeric fields."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def translate_dls_metadata(
    payload: dict[str, Any],
    label_type_map: dict[str, str] | None = None,
    timestamp_offset_ms: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """Convert a DLS inference metadata payload to Nx push format.

    Args:
        payload: Raw DLStreamer MQTT metadata dict.
        label_type_map: Optional mapping of detection label (lower-cased) to Nx
            typeId.  Labels absent from the map resolve to
            ``python.detected.object``.  Typically comes from
            ``ObjectDetectionAnalyticsAppConfig.label_type_map``.
        timestamp_offset_ms: Milliseconds added to the computed timestamp before
            pushing.  Use a negative value (e.g. ``-300``) to compensate for
            inference pipeline latency and align metadata with the video frame
            in Nx.  Configured via
            ``ObjectDetectionAnalyticsAppConfig.metadata_timestamp_offset_ms``.

    Returns a tuple of:
    - list of Nx object dicts (may be empty if no valid detections)
    - timestamp_ms to use for the metadata push

    The timestamp is taken from ``rtp.sender_ntp_unix_timestamp_ns`` (converted
    to milliseconds) when present, so the metadata aligns with the video frame
    in Nx.  Falls back to local wall-clock time when the field is absent.
    ``timestamp_offset_ms`` is applied in both cases.

    The ``typeId`` of each object is resolved from the detection label via
    ``label_type_map``, falling back to ``python.detected.object`` for
    unrecognised labels.
    """
    _map = {k.lower(): v for k, v in (label_type_map or {}).items()}
    wall_clock_ms = int(time.time() * 1000)
    rtp = payload.get("rtp") or {}
    ntp_ms = _ns_epoch_to_ms(rtp.get("sender_ntp_unix_timestamp_ns", 0))
    pipeline_time_ms = _ns_epoch_to_ms(payload.get("time", 0))

    # Prefer a payload-derived epoch timestamp when it is reasonably close to
    # current wall-clock time. If multiple are plausible, use the one with the
    # smallest skew. Fall back to wall-clock time if all payload clocks are stale.
    candidates = [ts for ts in (ntp_ms, pipeline_time_ms) if ts > 0]
    valid = [ts for ts in candidates if abs(wall_clock_ms - ts) <= _MAX_TIMESTAMP_SKEW_MS]
    if valid:
        base_timestamp_ms = min(valid, key=lambda ts: abs(wall_clock_ms - ts))
    else:
        base_timestamp_ms = wall_clock_ms

    timestamp_ms = base_timestamp_ms + timestamp_offset_ms

    objects: list[dict[str, Any]] = []
    for obj in payload.get("objects", []):
        detection = obj.get("detection") or {}
        bbox = detection.get("bounding_box") or {}

        x_min = bbox.get("x_min")
        y_min = bbox.get("y_min")
        x_max = bbox.get("x_max")
        y_max = bbox.get("y_max")
        if None in (x_min, y_min, x_max, y_max):
            continue

        width = max(0.0, x_max - x_min)
        height = max(0.0, y_max - y_min)
        # Nx bounding box format: "x,y,widthxheight" (all normalized 0–1)
        bounding_box = f"{x_min:.4f},{y_min:.4f},{width:.4f}x{height:.4f}"

        confidence = _to_float(detection.get("confidence"), 0.0)
        label = str(detection.get("label") or obj.get("roi_type") or "unknown").strip() or "unknown"

        object_seed = obj.get("id")
        region_id = obj.get("region_id")
        track_seed = object_seed if object_seed is not None else region_id

        # Prefer the DLS object id for stable tracking. Fall back to region_id,
        # then to a random UUID when neither identifier is present.
        track_id = str(uuid.UUID(int=int(track_seed))) if track_seed is not None else str(uuid.uuid4())

        type_id = _label_to_type_id(label, _map)

        attributes = [
            {"type": "String", "name": "label", "value": label, "confidence": confidence},
        ]

        objects.append(
            {
                "trackId": track_id,
                "typeId": type_id,
                "boundingBox": bounding_box,
                "confidence": confidence,
                "attributes": attributes,
            }
        )

    return objects, timestamp_ms
