"""Making team brand colours readable on a near-black page.

Several teams' official colours are effectively black (Iowa, Army, Cincinnati) or
extremely dark navy. Painted onto the dark detail panel those arcs and bars
disappear, so the chart looks broken rather than close. `readable` lifts anything
below a luminance floor toward white until it clears, keeping the hue intact, and
leaves everything already bright enough untouched.

Shared by the Jinja filter in app.py and the offline template tests. The same rule
is mirrored in the matchup predictor's JavaScript, which receives raw colours in
its JSON payload rather than through this filter.
"""

#How bright a colour's strongest channel has to be to read against the dark panel.
#Relative luminance is the wrong test here: it weights green heavily, so a vivid
#team red scores lower than a murky olive and gets "fixed" when it was already fine.
#The strongest channel tracks what the eye actually picks out on a dark ground.
CHANNEL_FLOOR = 0.45


def _parse_hex(value):
    text = str(value or "").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    if len(text) != 6:
        return None
    try:
        return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def _brightest(rgb):
    return max(rgb) / 255


def readable(value, fallback="#6b7684"):
    rgb = _parse_hex(value)
    if rgb is None:
        return fallback
    if _brightest(rgb) >= CHANNEL_FLOOR:
        return "#%02x%02x%02x" % rgb

    #Blend toward white by the smallest amount that clears the floor. Mixing with
    #white rather than scaling the channels keeps a near-black navy reading as navy
    #instead of collapsing to grey.
    for step in range(1, 21):
        t = step / 20
        mixed = tuple(round(channel + (255 - channel) * t) for channel in rgb)
        if _brightest(mixed) >= CHANNEL_FLOOR:
            return "#%02x%02x%02x" % mixed
    return fallback
