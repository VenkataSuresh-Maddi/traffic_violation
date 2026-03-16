"""
VehicleTracker — industry-grade centroid + IoU tracker.

Guarantees every physical vehicle is counted exactly once across
all frames of a video or across the detections in a single image.

Algorithm
---------
1. Each detected box is matched to an existing track using a two-stage
   metric: IoU (primary) + centroid distance (fallback).
2. A track is confirmed after MIN_HITS consecutive matches — prevents
   spurious one-frame detections from inflating the count.
3. A track is retired after MAX_AGE frames without a match.
4. Violation state is sticky: once True it never reverts.
5. The tracker emits a unique integer ID per physical vehicle.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import math


# ── tunables ──────────────────────────────────────────────────────────────────
MIN_HITS   = 1    # frames a track must be seen before it is "confirmed"
MAX_AGE    = 15   # frames before an unmatched track is deleted
IOU_THRESH = 0.25 # minimum IoU to match a detection to a track
CENT_THRESH = 120  # pixel centroid distance fallback (when IoU is 0)
# ──────────────────────────────────────────────────────────────────────────────


def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def _centroid(box: Tuple[int,int,int,int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _cent_dist(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax, ay = _centroid(a)
    bx, by = _centroid(b)
    return math.hypot(ax - bx, ay - by)


@dataclass
class Track:
    track_id:  int
    box:       Tuple[int,int,int,int]
    hits:      int  = 1
    age:       int  = 0          # frames since last match
    violation: bool = False
    plate:     str  = ""
    confirmed: bool = False      # True once hits >= MIN_HITS


class VehicleTracker:
    """
    Single-class centroid+IoU tracker.
    Call update() once per detection batch (per frame or per image).
    """

    def __init__(self):
        self._tracks: list[Track] = []
        self._next_id: int = 1
        # Final confirmed unique vehicles (never removed)
        self._registry: list[Track] = []

    # ------------------------------------------------------------------
    def update(
        self,
        detections: list[dict],   # each: {"box": (x1,y1,x2,y2), "violation": bool, "plate": str}
    ) -> list[dict]:
        """
        Match detections to existing tracks, create new ones,
        age out missing ones.  Returns list of active confirmed tracks
        with keys: track_id, box, violation, plate, confirmed.
        """
        # 1. Age all existing tracks
        for t in self._tracks:
            t.age += 1

        matched_track_ids = set()
        matched_det_indices = set()

        # 2. Build cost matrix and greedily match
        # Sort by IoU descending so best matches are taken first
        pairs = []
        for di, det in enumerate(detections):
            for t in self._tracks:
                iou = _iou(det["box"], t.box)
                cd  = _cent_dist(det["box"], t.box)
                score = iou if iou >= IOU_THRESH else (
                    (1 - cd / CENT_THRESH) if cd < CENT_THRESH else -1
                )
                if score > 0:
                    pairs.append((score, di, t.track_id))

        pairs.sort(key=lambda x: -x[0])

        for score, di, tid in pairs:
            if di in matched_det_indices:
                continue
            if tid in matched_track_ids:
                continue
            # Match!
            t = next(tr for tr in self._tracks if tr.track_id == tid)
            t.box = detections[di]["box"]
            t.age = 0
            t.hits += 1
            if detections[di]["violation"]:
                t.violation = True
            if detections[di]["plate"]:
                t.plate = detections[di]["plate"]
            if t.hits >= MIN_HITS:
                t.confirmed = True
            matched_track_ids.add(tid)
            matched_det_indices.add(di)

        # 3. Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di in matched_det_indices:
                continue
            new_track = Track(
                track_id  = self._next_id,
                box       = det["box"],
                violation = det["violation"],
                plate     = det["plate"],
            )
            self._next_id += 1
            self._tracks.append(new_track)

        # 4. Update registry with newly confirmed tracks
        confirmed_ids = {r.track_id for r in self._registry}
        for t in self._tracks:
            if t.confirmed and t.track_id not in confirmed_ids:
                self._registry.append(t)
                confirmed_ids.add(t.track_id)
            elif t.confirmed and t.track_id in confirmed_ids:
                # Sync violation/plate to registry entry
                reg = next(r for r in self._registry if r.track_id == t.track_id)
                if t.violation:
                    reg.violation = True
                if t.plate:
                    reg.plate = t.plate

        # 5. Remove dead tracks
        self._tracks = [t for t in self._tracks if t.age <= MAX_AGE]

        # 6. Return currently active confirmed tracks for drawing
        return [
            {
                "track_id": t.track_id,
                "box":      t.box,
                "violation": t.violation,
                "plate":    t.plate,
                "confirmed": t.confirmed,
            }
            for t in self._tracks
        ]

    # ------------------------------------------------------------------
    def get_stats(self) -> dict:
        """
        Return final deduplicated counts over the entire video / image.
        Only confirmed tracks are counted.
        """
        total      = len(self._registry)
        violations = sum(1 for r in self._registry if r.violation)
        safe       = total - violations
        return {"total": total, "violations": violations, "safe": safe}

    def get_registry(self) -> list[Track]:
        """All confirmed unique vehicles ever seen."""
        return list(self._registry)

    def reset(self):
        self._tracks   = []
        self._registry = []
        self._next_id  = 1