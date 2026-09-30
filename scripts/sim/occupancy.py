import bpy, sys, os, numpy as np, cv2
path, out = sys.argv[-2], sys.argv[-1]
skip_cats = set(filter(None, os.environ.get("OCC_SKIP_CATEGORIES", "").split(",")))   # e.g. door: treat door leaves as open
bpy.ops.wm.open_mainfile(filepath=path)
for o in bpy.data.objects:
    if skip_cats and o.get("hssd_category") in skip_cats:
        o.hide_render = True; o.hide_viewport = True
deps = bpy.context.evaluated_depsgraph_get()
rng = np.random.default_rng(0)
pts = []
for inst in deps.object_instances:
    o = inst.object
    if o.type != 'MESH': continue
    if skip_cats and o.get("hssd_category") in skip_cats: continue
    try: me = o.to_mesh()
    except Exception: continue
    if me is None or len(me.polygons) == 0:
        continue
    me.calc_loop_triangles()
    V = np.array([vv.co[:] for vv in me.vertices])
    T = np.array([t.vertices[:] for t in me.loop_triangles])
    a, b, c = V[T[:,0]], V[T[:,1]], V[T[:,2]]
    area = 0.5*np.linalg.norm(np.cross(b-a, c-a), axis=1)
    M = np.array(inst.matrix_world); S = np.abs(np.linalg.det(M[:3,:3]))**(2/3)
    n = np.maximum(1, np.ceil(area*S/0.0025)).astype(int)  # ~1 sample / 5cm^2
    n = np.minimum(n, 4000)
    idx = np.repeat(np.arange(len(T)), n)
    r1, r2 = rng.random(len(idx)), rng.random(len(idx))
    s = np.sqrt(r1)
    P = (1-s)[:,None]*a[idx] + (s*(1-r2))[:,None]*b[idx] + (s*r2)[:,None]*c[idx]
    pts.append((M[:3,:3] @ P.T).T + M[:3,3])
    o.to_mesh_clear()
P = np.concatenate(pts)
print("bbox", P.min(0).round(2), P.max(0).round(2), "n", len(P))
zmin, zmax = [float(v) for v in os.environ.get("OCC_Z", "0.25,1.7").split(",")]
sel = P[(P[:,2] > zmin) & (P[:,2] < zmax)]
res = 0.05
x0,y0 = P[:,0].min(), P[:,1].min()
W = int((P[:,0].max()-x0)/res)+1; H = int((P[:,1].max()-y0)/res)+1
grid = np.zeros((H,W), np.uint8)
ix = np.clip(((sel[:,0]-x0)/res).astype(int),0,W-1); iy = np.clip(((sel[:,1]-y0)/res).astype(int),0,H-1)
grid[iy, ix] = 255
np.savez(out.replace('.png','.npz'), grid=grid, x0=x0, y0=y0, res=res)
img = cv2.cvtColor(255-grid, cv2.COLOR_GRAY2BGR); img = cv2.flip(img, 0)
def py_(y): return H-1-int((y-y0)/res)
for x in range(int(np.floor(x0)), int(np.ceil(P[:,0].max()))+1):
    px = int((x-x0)/res); cv2.line(img,(px,0),(px,H-1),(0,0,255) if x==0 else (200,200,255),1); cv2.putText(img,str(x),(px+2,H-4),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,255),1)
for y in range(int(np.floor(y0)), int(np.ceil(P[:,1].max()))+1):
    py = py_(y); cv2.line(img,(0,py),(W-1,py),(0,0,255) if y==0 else (200,200,255),1); cv2.putText(img,str(y),(2,py-2),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,255),1)
s=max(1, 900//max(img.shape[:2])); img=cv2.resize(img,None,fx=s,fy=s,interpolation=cv2.INTER_NEAREST)
cv2.imwrite(out, img); print("saved", out)
