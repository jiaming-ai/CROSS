# CROSS website

The website of CROSS, open-source visual multi-session mapping, tracking and relocalization for robots, live at
<https://jiaming.im/CROSS/>. Code: [jiaming-ai/CROSS](https://github.com/jiaming-ai/CROSS). It is served by GitHub Pages
from the `gh-pages` branch of the code repository.

| path | page |
|---|---|
| `index.html` | the hub: what CROSS is, real runs, headline numbers, the benchmark ranking, the video, how it works, install, papers |
| `leaderboard/` | the benchmark at a glance: rankings by task, sensor and inputs, what is measured, datasets, systems |
| `benchmark/` | the full benchmark results, generated and published by the code repository (`benchmark/publish_page.sh`); never edit it here |
| `replays/` | animated trajectories from the benchmark: mapping one session, localizing another session in a stored map |
| `docs/` | documentation: install, quick start, the 14 sensor modes, own data, configuration, Python API, remote sessions |
| `v1/` | CROSS 1, the NeurIPS 2026 paper page (old links to `jiaming.im/CROSS/#<section>` redirect to it) |

Later paper pages (CROSS 2, CROSS 3, CROSS Cloud) go in their own folders (`v2/`, `v3/`, `cloud/`); add them to `PAGES` in
`tools/build.py`, to the papers section of `index.html` and to `tools/partials/footer.html`.

## Style

Swiss editorial, after [sgs-rl.github.io](https://sgs-rl.github.io/): Inter Tight at large sizes with tight tracking,
black rules across the window, a 12-column full-width grid, light statements with the key phrase in bold, square
corners, a red accent (yellow in the dark theme), full-bleed colour bands and halftone tiles. Tokens, the top bar and
the footer are in `assets/css/chrome.css` (every page loads it), components in `assets/css/site.css`, page styles in
`assets/css/<page>.css`.

## Build

The pages are static HTML; `tools/build.py` (standard library only) keeps their generated parts in sync:

```bash
python3 tools/build.py              # rebuild data, generated blocks, numbers, header/footer, sitemap
python3 tools/build.py --check      # exit 1 if anything is out of date
python3 tools/build.py --list-nums  # the named numbers pages can use (data-num="key")
```

- **Data**: `assets/data/leaderboard.json` and `assets/data/replays/*.json`, read from the published benchmark page
  (`benchmark/data.js`, `benchmark/runs/*.js`). After the code repository republishes the benchmark, run the build so the
  hub, the leaderboard and the replays agree with it.
- **Numbers**: any element with `data-num="key"` gets the value of `key` (e.g. `t3.cross.overall`).
- **Generated blocks**: `<!-- gen:NAME --> ... <!-- /gen:NAME -->` (the leaderboard tables, the hub's ranking).
- **Partials**: the shared header and footer (`tools/partials/*.html`) between `<!-- partial:NAME -->` markers.
- `sitemap.xml`.

Run it after editing any page. Social cards (`assets/img/og/*.jpg`): `bash tools/og/make_og.sh` (needs `google-chrome`
and ImageMagick). Halftone tiles: `python3 tools/halftone.py SRC OUT.png --crop x0 y0 x1 y1 --gamma 1.5`.

## Media

`assets/video/` holds web encodes of the CROSS showcase video (`showcase.mp4`, 5:09, 720p, with sound) and short
seamless clips cut from its parts (`clip-*.mp4`, 960x540, silent, the `.jpg` poster is the first frame). They were cut
from the renders of the video project (`/data2/projects/cross_v2/video/out/`) with ffmpeg on a server; clips play only
while on screen and never with reduced motion.

## Preview locally

```bash
python3 -m http.server 8000         # then http://localhost:8000/
```

Python's server has no HTTP range requests, so videos cannot seek there; GitHub Pages and `npx serve` support them.
Append `?theme=dark` or `?theme=light` to a URL to force a theme.

## Deploy

Commit on `gh-pages` and push; GitHub Pages rebuilds automatically (`.nojekyll` serves files as they are).
