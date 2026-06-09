"""
Tile per-song comparison sheets into paper-ready figures.

Expects evaluate_heldout.py and caption_baseline.py to have already produced
the individual images under --input-dir. Produces:

  paper/figures/fig_grid_ariana.png      4 songs x 6 columns
  paper/figures/fig_grid_drake.png       5 songs x 6 columns
  paper/figures/fig_grid_lesserafim.png  3 songs x 6 columns

Columns:  GT | CAPTION-ONLY | AUDIO-ONLY | STYLE | BIOGRAPHY | COMBINED
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLUMNS = [
    ("GT",        "{stem}_GT.png",                     "ground_truth"),
    ("CAPTION",   "{stem}_caption_combined_{steps}.png", "caption_baseline"),
    ("AUDIO ONLY","{stem}_audio_only_{steps}.png",     "trained"),
    ("STYLE",     "{stem}_style_{steps}.png",          "trained"),
    ("BIOGRAPHY", "{stem}_biography_{steps}.png",      "trained"),
    ("COMBINED",  "{stem}_combined_{steps}.png",       "trained"),
]

ARTIST_SONGS = {
    "ariana": ["stuckwithu", "wecantbefriends", "yesand", "problem"],
    "drake": ["inmyfeelings", "laughnowcrylater", "niceforwhat", "passionfruit", "themotto"],
    "lesserafim": ["antifragile", "perfectnight", "unforgiven"],
}

COVER_DIRS = {
    "ariana": "arianacovers",
    "drake": "drakecovers",
    "lesserafim": "lesserafimcovers",
}


def _find_gt(stem, artist):
    d = Path(COVER_DIRS[artist])
    for ext in (".jpg", ".jpeg", ".png"):
        p = d / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _load(path, size):
    if path is None or not Path(path).exists():
        return Image.new("RGB", (size, size), "lightgray")
    return Image.open(path).convert("RGB").resize((size, size))


def build_grid(artist, songs, input_dir, steps, cell, font):
    n_rows = len(songs)
    n_cols = len(COLUMNS)
    header_h = 36
    row_label_w = 130
    grid_w = row_label_w + n_cols * cell
    grid_h = header_h + n_rows * cell
    canvas = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(canvas)

    # column headers
    for c, (label, _, _) in enumerate(COLUMNS):
        x = row_label_w + c * cell
        box = draw.textbbox((0, 0), label, font=font)
        draw.text((x + (cell - (box[2] - box[0])) / 2, 8), label, fill="black", font=font)

    for r, stem in enumerate(songs):
        y = header_h + r * cell
        # row label
        draw.text((6, y + cell // 2 - 10), stem, fill="black", font=font)
        for c, (_, pattern, kind) in enumerate(COLUMNS):
            x = row_label_w + c * cell
            if kind == "ground_truth":
                img = _load(_find_gt(stem, artist), cell)
            else:
                fname = pattern.format(stem=stem, steps=steps)
                img = _load(Path(input_dir) / fname, cell)
            canvas.paste(img, (x, y))
            # thin border
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline="black", width=1)

    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="outputs/heldout_varied_seeds")
    ap.add_argument("--output-dir", default="paper/figures")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cell", type=int, default=192,
                    help="rendered cell size in px; smaller keeps paper figures tidy")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for artist, songs in ARTIST_SONGS.items():
        canvas = build_grid(artist, songs, args.input_dir, args.steps, args.cell, font)
        out = Path(args.output_dir) / f"fig_grid_{artist}.png"
        canvas.save(out)
        print(f"wrote {out}  ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
