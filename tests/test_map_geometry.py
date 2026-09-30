"""Persisted geometry must retain its declared inference semantics."""
import pytest

from cross.core.map_geometry import validate_load_mode
from cross.mono.config import MonoConfig


@pytest.mark.parametrize('saved',[
    {'schmidt_map_geometry_version':1},
    {'hypo_data':{'source_belief':{'keys':['image:a','geometry:old:1:0']}}},
])
def test_loading_shared_geometry_requires_schmidt_mode(saved):
    with pytest.raises(ValueError,match='enable schmidt_map_geometry'):
        validate_load_mode(saved,False)
    validate_load_mode(saved,True)


def test_existing_conditional_map_can_opt_into_geometry_on_next_commit():
    saved={'hypo_data':{'source_belief':{'keys':['image:a']}}}
    validate_load_mode(saved,False)
    validate_load_mode(saved,True)
    with pytest.raises(ValueError,match='Unsupported'):
        validate_load_mode({'schmidt_map_geometry_version':2},True)


def test_schmidt_configuration_requires_the_conditional_mapping_pipeline():
    with pytest.raises(ValueError,match='requires conditional sources'):
        MonoConfig(schmidt_map_geometry=True)
    config=MonoConfig(frontend='streaming_pnp',retrieval_pose='metric_pnp',
        conditional_sources=True,chart_aware=True,session_recovery=True,schmidt_map_geometry=True)
    assert config.trace_metric_sources
