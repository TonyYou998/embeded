"""Pixel-coordinate ratios. Left/right refer to the person's anatomy."""
from math import dist

LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)


def ear(points, indices):
    a, b, c, d, e, f = [points[i] for i in indices]
    width = dist(a, d)
    return (dist(b, f) + dist(c, e)) / (2 * width) if width else None


def ratios(landmarks, width, height):
    points = [(p.x * width, p.y * height) for p in landmarks]
    mouth_width = dist(points[78], points[308])
    # Defined here as inner-lip height / inner-lip width; no yawn threshold.
    mar = dist(points[13], points[14]) / mouth_width if mouth_width else None
    return {'ear_l': ear(points, LEFT_EYE), 'ear_r': ear(points, RIGHT_EYE), 'mar': mar}
