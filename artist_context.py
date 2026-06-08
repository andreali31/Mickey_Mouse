import json
from pathlib import Path


PROFILE_MODES = ("style", "biography", "combined")
DEFAULT_PROFILE_PATH = Path(__file__).with_name("artist_profiles.json")


def load_profiles(path=DEFAULT_PROFILE_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def profile_text(profiles, artist, mode):
    if mode not in PROFILE_MODES:
        raise ValueError(f"Unknown profile mode {mode!r}; choose from {PROFILE_MODES}")
    if artist not in profiles:
        raise KeyError(f"No profile for artist {artist!r}")
    profile = profiles[artist]
    if mode == "combined":
        details = f"{profile['biography']}. Musical and visual direction: {profile['style']}"
    else:
        details = profile[mode]
    return f"Album cover context for {profile['display_name']}: {details}"
