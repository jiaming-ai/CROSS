# Development split

A small part of the benchmark for testing a change in minutes before running the full benchmark. It follows the
[protocol](PROTOCOL.md) exactly: 10 Hz, the same odometry, thresholds (1 / 2 m indoors), 100-frame trials with stride 50,
coverage rule and metric code. Only the sequences differ: a few queries on which the CROSS modes disagree in the full
benchmark, some of them cut to their first frames. Dev results are never part of [RESULTS.md](RESULTS.md) or the
results page.

## Content

| tier | dataset | map → queries (frames) | why |
|---|---|---|---|
| quick | OpenLORIS office | office1-1 (269) → office1-2 (299), office1-7 (380), office1-4 (289) | motion-capture ground truth; RGB-D relocalizes in 0 of the T3 trials of office1-2/4/7, stereo in about half |
| quick | SimChange classroom | map (159) → light_night, move_50 (140 each) | controlled change of lighting and of object placement |
| full | OpenLORIS home | home1-1, first 600 frames → home1-4, first 400 | home furniture; stereo 14/14 vs RGB-D 8/14 trials in the full benchmark |
| full | OpenLORIS cafe | cafe1-1 (570) → cafe1-2, first 300 | people and clutter |
| full | KITTI 07 | 07 (1101), T1 only | outdoor scale and the loop closure at the end of the drive |

A tier `corridor` (entry `corridor_openloris`) is the benchmark's OpenLORIS corridor scene: map corridor1-1, its four
queries, T1 / T2 / T3. It checks the loop closures of a long, repetitive corridor, which the other tiers do not contain.

A third tier, `val`, is a confirmation set: every OpenLORIS office, home and cafe query of the benchmark (11 queries,
about 106 T3 trials), T2 and T3 only (entry `val_openloris`, `tier: val`). It takes about 25 minutes per stereo variant
on an RTX 5090, and is meant for a change that passed the dev split, before the full benchmark.

The full tier contains the quick tier. The entries are `dev_openloris`, `dev_simchange` and `dev_kitti` in
[configs/datasets.yaml](configs/datasets.yaml) (`dev: true`). They read the prepared folders of `openloris`,
`simchange` and `kitti` (`data:`), so no data is duplicated. Only the map sequences get a T1 result.

One CROSS mode takes about 7 minutes for the quick tier and 14 minutes for the full tier on an RTX 5090
(stereo mode, alone on the GPU).

## Use

```bash
export BENCH_DATA=/path/to/benchmark_data BENCH_DEV_RESULTS=/path/to/dev_runs
python benchmark/datasets/make_dev.py                 # once: the clipped sequences of the full tier (symlinks)

# reference with the current code, then a variant (a label; --args are added to the CROSS command)
python benchmark/dev.py run --systems cross_stereo --tier quick --gpus 0
python benchmark/dev.py run --systems cross_stereo --tier quick --gpus 0 --variant views7 --args "--max-refs 4 --n-ref-anchors 1"
python benchmark/dev.py compare cross_stereo cross_stereo@views7
```

`run` builds the maps first, then runs the query jobs (T2 and T3), one job per GPU slot (`--gpus 0 1 2`). Jobs whose
result exists are skipped. Results go to `$BENCH_DEV_RESULTS/<dataset>/<scene>/<system>[@<variant>]/`. A change of
code (not of arguments) is tested by running the reference before the change and a variant after it.

`compare` prints every cell side by side: T1 ATE and FPS, T2 recall at both thresholds per query, T3 successes per
query with **paired flips** against the first run (`+won-lost` on the same trial starts), the pooled numbers, and
the GPU time of the jobs.

## Reading a comparison

- **T2 recall** (hundreds of frames per query) and T1 ATE are the regression signals. The quick tier has 16 T3 trials
  and the full tier 27, so a T3 rate alone cannot detect a change of less than about 20 %. The paired flips say which trials
  changed.
- **Noise.** On an otherwise idle GPU, two runs of the same code are identical (checked on the RTX 5090, 2026-10-01),
  so a code change that should not change results can be checked exactly. That does not make every difference
  meaningful: a change that should be neutral (transformer weights in fp32 instead of bf16) moved office1-7 T2 LR@1
  from 0.45 to 0.86 and flipped one T3 trial, because one continuous run is sensitive to small perturbations. Treat
  single-query T2 swings and single T3 flips as noise; run such a null variant next to a real one when in doubt
  (`--variant fp32w --args "--set pose_est.ff.half_precision_weights=false"` for the stereo mode). On a shared GPU, or
  with a different GPU model, results also vary, so compare only runs from the same GPU; `compare` warns when the GPUs
  differ.
- The SimChange classroom queries are easy for every stereo variant tried (one T3 trial each, same T2); they guard
  against breakage rather than separate variants. The same holds for the full tier's home and cafe queries in the
  stereo mode.
- A change that passes the dev split is then run on the full benchmark (`benchmark/README.md`).
