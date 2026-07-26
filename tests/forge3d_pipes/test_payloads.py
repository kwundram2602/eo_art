import json
from pathlib import Path

from omegaconf import OmegaConf

from eo_art.forge3d_pipes.config.schema import PipelineConfig
from eo_art.forge3d_pipes.render import payloads

GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "golden_payloads.json").read_text()
)


def _default_cfg(**overrides) -> PipelineConfig:
    merged = OmegaConf.merge(
        OmegaConf.structured(PipelineConfig),
        {"input": {"path": "dem.tif"}},
        OmegaConf.from_dotlist([f"{k}={v}" for k, v in overrides.items()]),
    )
    return OmegaConf.to_object(merged)


def test_set_terrain_matches_demo1_golden():
    assert payloads.build_set_terrain(_default_cfg()) == GOLDEN["set_terrain"]


def test_set_terrain_pbr_matches_demo1_golden():
    assert payloads.build_set_terrain_pbr(_default_cfg()) == GOLDEN["set_terrain_pbr"]


def test_enums_serialise_to_their_string_values():
    pbr = payloads.build_set_terrain_pbr(_default_cfg())
    assert pbr["shadow_technique"] == "pcss"
    assert pbr["tonemap"]["operator"] == "aces"
    assert pbr["sun_visibility"]["mode"] == "soft"


def test_overrides_reach_the_payload():
    cfg = _default_cfg(
        **{
            "render.camera.phi": 42.0,
            "render.pbr.exposure": 2.0,
            "render.pbr.sky.turbidity": 9.0,
        }
    )
    assert payloads.build_set_terrain(cfg)["phi"] == 42.0
    pbr = payloads.build_set_terrain_pbr(cfg)
    assert pbr["exposure"] == 2.0
    assert pbr["sky"]["turbidity"] == 9.0


def test_payloads_are_plain_json_serialisable():
    cfg = _default_cfg()
    json.dumps(payloads.build_set_terrain(cfg))
    json.dumps(payloads.build_set_terrain_pbr(cfg))
