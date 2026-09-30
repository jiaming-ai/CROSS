#!/usr/bin/env bash
# ORB-SLAM3 stereo mapping of the HSSD house at three pacing values (ms of sleep per frame): does the atlas
# fragmentation (tracking loss at the rotation stations) depend on how fast the frames are fed?
cd "$(dirname "$0")/../.."
ROOT=$PWD; ORB=$ROOT/third_party/ORB_SLAM3
export LD_LIBRARY_PATH=$ROOT/third_party/install/lib:$ORB/lib:$ORB/Thirdparty/DBoW2/lib:$ORB/Thirdparty/g2o/lib:$LD_LIBRARY_PATH
M=$ROOT/data/sim/hssd_house/map
RD=$(.venv/bin/python -c "import json;print(json.load(open('$M/calib.json'))['right_dirs']['0.30'])")
for P in 5 100 300; do
  O=$ROOT/outputs/orb_pace/hssd_house_p$P; mkdir -p $O; cd $O
  .venv/bin/python - <<PY 2>/dev/null || true
PY
  $ROOT/.venv/bin/python -c "
import sys; sys.path.insert(0,'$ROOT/scripts/baselines'); sys.path.insert(0,'$ROOT/scripts'); sys.path.insert(0,'$ROOT')
from pathlib import Path; import run_baselines as rb
rb.orb_yaml(Path('$M'), Path('$O/map.yaml'), save_atlas='atlas', fps=10.0, baseline=0.3)"
  /usr/bin/time -f "%e s" $ROOT/scripts/baselines/build/orbslam3_reloc $ORB/Vocabulary/ORBvoc.txt $O/map.yaml $M $O/map_poses.txt --fps 10 --right-dir $RD --pace-ms $P > $O/map.log 2>&1
  echo "pace $P: nmaps=$(cat $O/map_poses.txt.final_nmaps) states: $(awk '{print $2}' $O/map_poses.txt | sort | uniq -c | tr '\n' ' ') largest-map frames: $(awk '{print $9}' $O/map_poses.txt.final | sort | uniq -c | sort -rn | head -1)" | tee -a $ROOT/logs/orb_pace.log
done
echo ORB_PACE_DONE >> $ROOT/logs/orb_pace.log
