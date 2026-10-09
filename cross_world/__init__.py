"""CROSS World: a 3D Gaussian-splatting reconstruction of a CROSS map for novel view synthesis.

A CROSS map is a topological graph whose permanent nodes are posed keyframe images (RGB-D, stereo or monocular).  This
package turns a saved map into a radiance field (3D Gaussians) that can be rendered from any viewpoint:

    map_views    the map's keyframes as posed views (pose, intrinsics, image, depth, right image), optionally at the
                 full resolution of the source sequence, and its covisibility graph
    depth        per-view metric depth: sensor depth, stereo matching of the stored pair, or learned depth
    partition    chunks of a large map (balanced spatial split of the keyframes, with overlap) trained independently
    gaussians    initialisation from back-projected depth, training (gsplat), keyframe anchoring
    world        the reconstruction of a whole map: build, save / load, re-pose after the map's poses change
    export       PLY / SPZ files and the data of the web viewer

Entry point: `python -m cross_world.cli --help`.
"""

__version__ = "0.1.0"
