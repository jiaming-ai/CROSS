# CROSS project page

Project page for *CROSS: Change-Robust Online Topological Memory for Long-Term Relocalization and Semantic Navigation* (NeurIPS 2026),
live at <https://jiaming.im/CROSS/>. Paper: [arXiv:2605.02227](https://arxiv.org/abs/2605.02227). Code: [jiaming-ai/CROSS](https://github.com/jiaming-ai/CROSS).

It is a static site (plain HTML, CSS and JavaScript, no build step), served by GitHub Pages from the `gh-pages` branch of the code repository.

## Preview locally

```bash
npx serve .                  # or: python3 -m http.server 8000
```

Python's built-in server does not support HTTP range requests, so the video chapter buttons cannot seek there.
`npx serve` and GitHub Pages both support them.

## Deploy

Commit and push to the `gh-pages` branch; GitHub Pages rebuilds automatically. `.nojekyll` is included so files are served as is.

## Layout

```
index.html              page content
static/css/style.css    styles (light and dark themes)
static/js/main.js       page behaviour and chart data (numbers from the paper's tables)
static/js/charts.js     small SVG chart helpers
static/js/sim.js        toy 2-D simulation behind the "See the difference" demo
static/js/demo.js       renders the demo
static/img/, static/video/  figures and videos
```

`static/video/cross.mp4` is a web re-encode (H.264, CRF 27, 27 MB) of the full video;
`static/video/hero.mp4` is the first 11 seconds at 720p, used as the header background.
