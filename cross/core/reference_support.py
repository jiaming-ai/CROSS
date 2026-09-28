"""Direct historical evidence for delayed recovery of a loaded map.

A new session's arbitrary chart offset is not a physical separation from the
loaded map. This bookkeeping can replace the distance guard for such a
candidate, but cannot replace CROSS's temporal overlap/confidence tests.
"""
from collections import deque


class ReferenceSupport:
    def __init__(self, components, window, hit_rate, enabled=False):
        self.components, self.window, self.hit_rate = components, window, hit_rate
        self.enabled = enabled
        self.loaded_ids = frozenset()
        self.unanchored = False
        self.clear_tracking()

    def clear_tracking(self):
        self.history = [deque([frozenset()] * self.window, maxlen=self.window)
                        for _ in range(self.components)]
        self.ever_historical = [False] * self.components
        self.last_step = None

    def start(self, loaded_ids):
        self.loaded_ids = frozenset(loaded_ids)
        self.unanchored = self.enabled and bool(self.loaded_ids)
        self.clear_tracking()

    def observe(self, step, edge_mapping, newborns=()):
        if not self.unanchored:
            return
        if self.last_step == step:
            return  # one source observation cannot count twice
        if self.last_step is not None and step < self.last_step:
            raise ValueError("Reference support requires increasing source steps")
        self.last_step = step
        for component in newborns:
            self.history[component] = deque([frozenset()] * self.window, maxlen=self.window)
            self.ever_historical[component] = False
        supports = [set() for _ in range(self.components)]
        for keyframe, (source_component, target_component) in edge_mapping.items():
            # Only a directly verified loaded-map pose is independent of
            # query nodes that already inherited this candidate's belief.
            if keyframe in self.loaded_ids and source_component == 0:
                supports[target_component].add(keyframe)
        for component, ids in enumerate(supports):
            self.history[component].append(frozenset(ids))
            self.ever_historical[component] |= bool(ids)

    def audit(self, component):
        history = self.history[component]
        hits = sum(bool(ids) for ids in history)
        references = sorted(set().union(*history))
        cross_chart = self.unanchored and self.ever_historical[component]
        # Use the inherited overlap hit-rate/window and require redundancy
        # across at least two reference images. Neither is place proof alone.
        eligible = cross_chart and hits / self.window >= self.hit_rate and len(references) >= 2
        return dict(unanchored_reference_candidate=bool(cross_chart),
                    supported_frames=hits, window=self.window, reference_ids=references,
                    eligible=bool(eligible))

    def mark_anchored(self):
        self.unanchored = False
