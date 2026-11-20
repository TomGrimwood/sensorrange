"""What the check pictures share: their colours, fonts and words for angles."""

from __future__ import annotations

from PIL import ImageFont

MUTED = (110, 112, 108)
BLUE, RED, INK, GRID, PAPER = (
    (20, 90, 200),
    (220, 40, 40),
    (30, 30, 30),
    (215, 215, 210),
    (250, 250, 247),
)


def _side(degrees: float, digits: int = 1) -> str:
    if not round(degrees, digits):
        return "ahead"
    return f"{abs(degrees):.{digits}f}° {'left' if degrees > 0 else 'right'}"


def _updown(degrees: float, digits: int = 1) -> str:
    if not round(degrees, digits):
        return "level"
    return f"{abs(degrees):.{digits}f}° {'up' if degrees > 0 else 'down'}"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()
