"""Closed-loop point-goal evaluation and DAgger data collection in Habitat (HSSD scenes).

geom.py     robot-frame conversions, path resampling, causal history selection, waypoint controller (numpy only)
wire.py     numpy-safe messages between the simulator processes and the GPU policy server
episodes.py navmesh episode sampling per HSSD split (habitat env)
rollout.py  simulator worker: expert / policy / DAgger rollouts, metrics, optional recording (habitat env)

The simulator side never imports torch; the policy server (cross_nav.policy.server) runs in the nav environment.
"""
