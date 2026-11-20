"""AprilTags a camera saw against where the range says they are: each tag36h11 tag OpenCV
finds in a frame, its corners against its corners on record in the range's USD drawn through the camera's
lens and pose."""

from __future__ import annotations

import cv2
import numpy as np

from .ranges import Tag


def check(
    grey: np.ndarray, lens: dict, rotation: np.ndarray, at: np.ndarray, tags: tuple[Tag, ...]
) -> list[tuple[str, str]]:
    """Rows for a camera's table: how many of `tags` are in its view (every corner in its
    frame), how many OpenCV found, those it missed, and how far found corners are from where
    its lens and pose put them (pixels)."""
    from .sheets import project

    w, h = lens["width"], lens["height"]
    expected = {}
    for tag in tags:
        top_left, top_right, _, bottom_left = tag.corners
        front = np.cross(bottom_left - top_left, top_right - top_left)  # out of its face
        if front @ (at - tag.corners.mean(axis=0)) <= 0:
            continue  # it faces away
        pixels = project((tag.corners - at) @ rotation, lens)
        u, v = pixels[:, 0], pixels[:, 1]
        if (
            np.isfinite(pixels).all()
            and (u >= 0).all()
            and (u < w).all()
            and (v >= 0).all()
            and (v < h).all()
        ):
            expected[tag.id] = pixels
    settings = cv2.aruco.DetectorParameters()
    settings.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    finder = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), settings
    )
    corners, ids, _ = finder.detectMarkers(grey.astype(np.uint8))
    # OpenCV's pixel centres are whole numbers; `project`'s are halves (pixel i spans i to i + 1)
    found = (
        {int(i): c.reshape(4, 2) + 0.5 for i, c in zip(ids.ravel(), corners, strict=True)}
        if ids is not None
        else {}
    )
    matched = [i for i in found if i in expected]
    off = (
        np.concatenate([np.linalg.norm(found[i] - expected[i], axis=1) for i in matched])
        if matched
        else np.zeros(0)
    )
    missed = sorted(set(expected) - set(found))
    rows = [("tags in view", str(len(expected))), ("tags found", str(len(matched)))]
    if missed:
        rows.append(("missed", _ids(missed)))
    if len(off):
        rows.append(("corner error", f"{off.mean():.2f} px"))
        rows.append(("worst corner", f"{off.max():.2f} px"))
    return rows


def _ids(ids: list[int]) -> str:
    """Ids as runs: 30-33, 40."""
    runs, start = [], ids[0]
    for a, b in zip(ids, [*ids[1:], None], strict=True):
        if b != a + 1:
            runs.append(f"{start}" if start == a else f"{start}-{a}")
            start = b
    return ", ".join(runs)
