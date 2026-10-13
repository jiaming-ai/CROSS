"""Convert ROS1 navigation bags (SCAND, TartanDrive) to realnav episodes.

    python -m cross_nav.policy.datasets.rosbag_nav --dataset scand --raw RAW_DIR --out REALNAV [--workers 12] [--wait]

RAW_DIR is searched recursively for *.bag. A bag is converted once its download is verified (a "<bag>.ok" file
from cross_nav.policy.datasets.fetch exists, or --no-ok-marker). One pass per bag: the camera stream is paired with
the odometry interpolated at each image's bag time (as in the GNM process_bags.py pipeline, with GNM's topics and
yaw offsets), frames are subsampled by travelled distance (MIN_STEP..MAX_STEP m, or a > 5 deg turn), and episodes are
cut at pose jumps (> 1.5 m), time gaps, odometry dropouts and backward driving (GNM also removes backward motion).
Positions are the odometry frame's x, y; yaw is the heading of the body x axis (forward) in that frame.
Only numeric arrays and encoded images are read from the bags.
"""
import argparse, bisect, io, json, math, re, shutil, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from .realnav_common import write_ready, previews, wrap

AZURE_KINECT = {'model': 'Azure Kinect RGB (per the SCAND README); no CameraInfo in the bags', 'fisheye': False,
                'resolution_native': [1280, 720], 'fov_deg_nominal': {'h': 90, 'v': 59},
                'note': 'nominal Azure Kinect 1280x720 colour-mode FoV from the sensor spec, not a calibration'}
PROFILES = {
    'scand': dict(
        # GNM's two SCAND topics, plus the stereo-left stream that replaces the Kinect in some Jackal bags
        image_topics=['/camera/rgb/image_raw/compressed', '/image_raw/compressed', '/left/image_color/compressed'],
        odom_topics=['/jackal_velocity_controller/odom', '/odom'], ang_offset=0.0,
        min_step=0.2, max_step=0.3, max_gap=1.0, min_arc=4.0,
        license='CC BY 4.0 (SCAND, Texas Data Repository doi:10.18738/T8/0PRYRH)',
        source='https://huggingface.co/datasets/franciszzj/SCAND (mirror of the Texas Data Repository release)',
        label_source='odometry',
        camera=lambda bag, topic: dict(
            AZURE_KINECT if topic != '/left/image_color/compressed' else
            {'model': 'left camera of a front stereo pair (frame left_camera); model and calibration not in the bag',
             'fisheye': False, 'resolution_native': [1280, 1024]},
            robot='Clearpath Jackal' if '_Jackal_' in bag else 'Boston Dynamics Spot',
            mount='front, ~0.5 m (Jackal) / ~0.8 m (Spot) above ground (approximate, not documented in the bags)'),
        odom_note=lambda bag: ('Jackal wheel odometry (/jackal_velocity_controller/odom)' if '_Jackal_' in bag
                               else 'Spot kinematic odometry (/odom, body frame)'),
        group=lambda bag: re.sub(r'_\d+$', '', bag),  # robot, route, recording day: e.g. A_Jackal_AHG_Library_Thu_Oct_28
    ),
    'tartan_drive': dict(
        image_topics=['/multisense/left/image_rect_color'], odom_topics=['/odometry/filtered_odom'],
        ang_offset=math.pi / 2,  # GNM process_bags_config.yaml: tartan_drive ang_offset pi/2
        min_step=0.2, max_step=0.3, max_gap=1.0, min_arc=8.0,
        license='TartanDrive 1.0 (CMU AirLab); tools repo castacks/tartan_drive is MIT, no separate data licence found',
        source='https://huggingface.co/datasets/franciszzj/TartanDrive_{0,1,2} (mirror of the TartanDrive 1.0 bags)',
        label_source='odometry',
        camera=lambda bag, topic: {'model': 'Carnegie Robotics MultiSense S21 stereo, left rectified colour', 'fisheye': False,
                            'robot': 'Yamaha Viking ATV (off-road)', 'mount': 'roof, ~2 m above ground (approximate)',
                            'note': 'no CameraInfo in the bags; rectified pinhole image'},
        odom_note=lambda bag: 'filtered odometry (/odometry/filtered_odom: GPS/INS + wheel EKF)',
        group=lambda bag: bag.split('_')[0],  # recording day, e.g. 20210903
    ),
}


def quat_yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def decode(msg, msgtype):
    if msgtype.endswith('CompressedImage'):
        return Image.open(io.BytesIO(bytes(msg.data))).convert('RGB')
    enc = msg.encoding.lower(); h, w = msg.height, msg.width
    a = np.frombuffer(bytes(msg.data), np.uint8)
    if enc in ('rgb8', 'bgr8'):
        a = a.reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3)
        return Image.fromarray(a[..., ::-1] if enc == 'bgr8' else a)
    if enc in ('rgba8', 'bgra8'):
        a = a.reshape(h, msg.step)[:, :w * 4].reshape(h, w, 4)[..., :3]
        return Image.fromarray(a[..., ::-1] if enc == 'bgra8' else a)
    if enc in ('mono8', '8uc1'):
        return Image.fromarray(a.reshape(h, msg.step)[:, :w]).convert('RGB')
    raise ValueError(f'unsupported image encoding {enc}')


class Writer:
    """Accumulates kept frames of the current episode and writes it when it ends."""

    def __init__(self, out, bag, prof, meta):
        self.out, self.bag, self.prof, self.meta = Path(out), bag, prof, meta
        self.k = 0; self.reset(); self.episodes = []

    def reset(self):
        self.dir = self.out / f'{self.bag}__{self.k:03d}'
        self.pos, self.yaw, self.t = [], [], []

    def add(self, img, pos, yaw, t):
        if not self.pos:
            if self.dir.exists(): shutil.rmtree(self.dir)
            self.dir.mkdir(parents=True)
        img = img.convert('RGB')
        if max(img.size) > 1024:
            s = 1024 / max(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
        img.save(self.dir / f'{len(self.pos)}.jpg', quality=95)
        self.size = img.size
        self.pos.append(pos); self.yaw.append(yaw); self.t.append(t)

    def close(self):
        if self.pos:
            p = np.asarray(self.pos, np.float64)
            arc = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0
            if len(p) >= 10 and arc >= self.prof['min_arc']:
                np.savez(self.dir / 'trajectory.npz', position=p, yaw=np.asarray(self.yaw, np.float64),
                         t=np.asarray(self.t, np.float64) - self.t[0])
                src = dict(self.meta, episode_start_bag_time=self.t[0], image_size=list(self.size))
                (self.dir / 'source.json').write_text(json.dumps(src, indent=1))
                self.episodes.append((self.dir.name, len(p), arc)); self.k += 1
            else:
                shutil.rmtree(self.dir)
        self.reset()


def convert_bag(path, out, dataset):
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)
    prof = PROFILES[dataset]; path = Path(path); bag = path.stem
    t0 = time.time()
    with Reader(path) as r:
        topics = {c.topic: c for c in r.connections}
        itop = next((t for t in prof['image_topics'] if t in topics), None)
        otop = next((t for t in prof['odom_topics'] if t in topics), None)
        if itop is None or otop is None:
            return dict(bag=bag, error=f'missing topics (image {itop}, odom {otop})', topics=sorted(topics))
        ic, oc = topics[itop], topics[otop]
        meta = dict(original=f'{path.name}:{itop}', group=prof['group'](bag), metric=True, label_source=prof['label_source'],
                    odometry=f"{prof['odom_note'](bag)}; topic {otop}, yaw offset {prof['ang_offset']:.4f} rad",
                    camera=prof['camera'](bag, itop), license=prof['license'], source=prof['source'], dataset=dataset,
                    fps=None, subsampling=f"kept frames {prof['min_step']}..{prof['max_step']} m apart (or > 5 deg turn)")
        n_img = ic.msgcount; dur = r.duration / 1e9
        meta['fps'] = round(n_img / dur, 2) if dur > 0 else None
        w = Writer(out, bag, prof, meta)
        ot, ox, oy, oyaw = [], [], [], []
        pending = []  # (t, raw) images waiting for odometry after their time
        st = dict(last=None, prev=None, last_t=None)  # last kept (pos, yaw, t); previous candidate (img raw, pos, yaw, t, kept?)
        cnt = dict(images=0, kept=0, cuts_jump=0, cuts_gap=0, cuts_back=0, no_odom=0)

        def pose_at(t):
            j = bisect.bisect_left(ot, t)
            if j == 0 or j >= len(ot): return None
            if ot[j] - ot[j - 1] > 0.5: return None  # odometry dropout
            a = (t - ot[j - 1]) / max(ot[j] - ot[j - 1], 1e-9)
            x = ox[j - 1] + a * (ox[j] - ox[j - 1]); y = oy[j - 1] + a * (oy[j] - oy[j - 1])
            yaw = oyaw[j - 1] + a * wrap(oyaw[j] - oyaw[j - 1])
            return np.array([x, y]), float(wrap(yaw + prof['ang_offset']))

        def keep(raw, pos, yaw, t):
            w.add(decode(ts.deserialize_ros1(raw, ic.msgtype), ic.msgtype), pos, yaw, t)
            st['last'] = (pos, yaw, t); cnt['kept'] += 1

        def handle(t, raw):
            cnt['images'] += 1
            pz = pose_at(t)
            if pz is None:
                cnt['no_odom'] += 1; w.close(); st['last'] = None; st['prev'] = None; return
            pos, yaw = pz
            last = st['last']
            if last is not None:
                d = pos - last[0]; dist = float(np.hypot(*d))
                if dist > 1.5 or t - last[2] > prof['max_gap'] and dist > prof['max_step']:
                    cnt['cuts_jump' if dist > 1.5 else 'cuts_gap'] += 1
                    w.close(); last = st['last'] = None; st['prev'] = None
            if last is None:
                keep(raw, pos, yaw, t); st['prev'] = None; return
            d = pos - last[0]; dist = float(np.hypot(*d)); turn = abs(float(wrap(yaw - last[1])))
            prev = st['prev']
            # keep the previous candidate if skipping it would leave a > max_step hole
            if dist > prof['max_step'] and prev is not None and float(np.hypot(*(prev[1] - last[0]))) >= 0.05:
                if not backward(last, prev[1]):
                    keep(prev[0], prev[1], prev[2], prev[3]); last = st['last']
                    d = pos - last[0]; dist = float(np.hypot(*d)); turn = abs(float(wrap(yaw - last[1])))
            if dist >= prof['min_step'] or turn > math.radians(5) or dist > prof['max_step']:
                if dist >= 0.05 and backward(last, pos):
                    cnt['cuts_back'] += 1; w.close(); st['last'] = None; st['prev'] = None
                    keep(raw, pos, yaw, t); return
                keep(raw, pos, yaw, t); st['prev'] = None
            else:
                st['prev'] = (raw, pos, yaw, t)

        def backward(last, pos):
            d = pos - last[0]
            return math.cos(math.atan2(d[1], d[0]) - last[1]) < -0.5  # moving > 120 deg away from the heading

        for c, t, raw in r.messages(connections=[ic, oc]):
            t = t / 1e9
            if c.id == oc.id:
                m = ts.deserialize_ros1(raw, c.msgtype)
                p, q = m.pose.pose.position, m.pose.pose.orientation
                if not all(map(math.isfinite, (p.x, p.y, q.w, q.x, q.y, q.z))): continue
                if ot and t <= ot[-1]: continue
                ot.append(t); ox.append(p.x); oy.append(p.y); oyaw.append(quat_yaw(q))
                while pending and pending[0][0] <= ot[-1]:
                    handle(*pending.pop(0))
            else:
                pending.append((t, raw))
        w.close()
    return dict(bag=bag, image_topic=itop, odom_topic=otop, duration_s=dur, images=cnt['images'], kept=cnt['kept'],
                cuts=dict(jump=cnt['cuts_jump'], gap=cnt['cuts_gap'], backward=cnt['cuts_back'], no_odom=cnt['no_odom']),
                episodes=len(w.episodes), arc_m=round(sum(e[2] for e in w.episodes), 1), seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=sorted(PROFILES)); ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--wait', action='store_true', help='keep polling for newly verified bags until --expect bags are done')
    ap.add_argument('--expect', type=int, default=0); ap.add_argument('--no-ok-marker', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='convert at most this many bags (testing)')
    a = ap.parse_args()
    out = Path(a.out) / a.dataset; out.mkdir(parents=True, exist_ok=True)
    log_path = out / 'conversion_log.jsonl'
    done = set()
    if log_path.exists():  # bags that failed are retried (their log lines are dropped)
        rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        rows = [r for r in rows if 'error' not in r]
        log_path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        done = {r['bag'] for r in rows}
    with ProcessPoolExecutor(a.workers) as ex:
        while True:
            bags = sorted(Path(a.raw).rglob('*.bag'))
            ready = [b for b in bags if b.stem not in done and (a.no_ok_marker or Path(str(b) + '.ok').exists())]
            if a.limit: ready = ready[:max(0, a.limit - len(done))]
            futs = {ex.submit(convert_bag, b, out, a.dataset): b for b in ready}
            for f in as_completed(futs):
                try: res = f.result()
                except Exception as e: res = dict(bag=futs[f].stem, error=repr(e))
                done.add(res['bag'])
                with open(log_path, 'a') as fh: fh.write(json.dumps(res) + '\n')
                print(json.dumps(res), flush=True)
            if not a.wait or len(done) >= a.expect or (a.limit and len(done) >= a.limit): break
            if not ready: time.sleep(60)
    finalize(out, a.dataset)


def finalize(out, dataset):
    prof = PROFILES[dataset]
    logs = [json.loads(l) for l in (out / 'conversion_log.jsonl').read_text().splitlines() if l.strip()]
    errors = [l for l in logs if 'error' in l]
    extra = dict(dataset=dataset, label_source=prof['label_source'], license=prof['license'], source=prof['source'],
                 bags_converted=len(logs) - len(errors), bags_failed=[(l['bag'], l['error']) for l in errors],
                 source_hours=round(sum(l.get('duration_s', 0) for l in logs) / 3600, 2),
                 converter=f'python -m cross_nav.policy.datasets.rosbag_nav --dataset {dataset}')
    ready = write_ready(out, extra)
    previews(out)
    print(json.dumps({k: v for k, v in ready.items() if k != 'bags_failed'}, indent=1), flush=True)


if __name__ == '__main__':
    import sys
    if sys.argv[1:2] == ['--finalize']:  # rewrite READY.json/previews of a finished conversion: --finalize OUT DATASET
        finalize(Path(sys.argv[2]) / sys.argv[3], sys.argv[3])
    else:
        main()
