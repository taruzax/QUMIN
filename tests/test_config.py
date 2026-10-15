from pathlib import Path

import pytest

import lab.core.config as config_module
from lab.core.config import CampaignMatrixConfig, IngestionConfig, PipelineConfig, PlatformConfig


def test_nested_config_round_trips_with_hash():
    config = PipelineConfig.from_yaml(Path("config/pipeline.yaml"))
    restored = PipelineConfig.model_validate(config.model_dump())
    assert restored.model_dump() == config.model_dump()
    assert restored.model_hash() == config.model_hash()
    assert config.models.xgboost.params["max_depth"] == 2
    assert config.models.early_stopping_fraction == 0.20
    assert config.models.early_stopping_patience == 3
    assert config.allocation.lookback_bars == 252


def test_all_bundled_yaml_config_readers_load_their_explicit_files():
    assert PipelineConfig.from_yaml("config/pipeline.yaml").data.source == "provider"
    assert PipelineConfig.from_yaml("config/demo.yaml").data.source == "bundled_demo"
    assert IngestionConfig.from_yaml("config/ingestion.yaml").streams
    assert PlatformConfig.from_yaml("config/platform.yaml").paths.artifacts_dir
    assert CampaignMatrixConfig.from_yaml("config/campaign.example.yaml").name


def test_config_accepts_string_yaml_path():
    config = PipelineConfig.from_yaml("config/pipeline.yaml")

    assert config.data.source == "provider"


def test_legacy_sections_translate_with_flat_compatibility_accessors():
    with pytest.warns(UserWarning):
        config = PipelineConfig(
            data={"timeframe": "1d"},
            returns={"target_horizons": [1]},
            labeling={"vol_lookback_bars": 12},
            risk={"max_position_size": 0.1},
            cv={"n_splits": 2},
        )
    assert config.timeframe.value == "1d"
    assert config.target_horizons == [1]
    assert config.vol_lookback_bars == 12
    assert config.max_position_size == 0.1
    assert config.n_splits == 2


def test_conflicting_legacy_and_canonical_settings_raise():
    with pytest.raises(ValueError, match="Conflicting settings"):
        PipelineConfig(
            allocation={"max_position_size": 0.2},
            risk={"max_position_size": 0.1},
        )


def test_unknown_nested_keys_are_rejected():
    with pytest.raises(ValueError):
        PipelineConfig(models={"xgboost": {"enabled": True, "misspelled": 1}})


def test_explicit_config_uses_only_selected_yaml_and_runtime_override_wins(tmp_path, monkeypatch):
    selected = tmp_path / "selected.yaml"
    selected.write_text("data:\n  source: local\n  timeframe: 1d\n")
    monkeypatch.setenv("DATA__TIMEFRAME", "1h")
    monkeypatch.setenv("MODELS__SEED", "81")

    config = PipelineConfig.from_yaml(selected, overrides={"data": {"timeframe": "1d"}, "models": {"seed": 91}})

    assert config.data.source == "local"
    assert config.data.timeframe.value == "1d"
    assert config.models.seed == 91
    assert config.source_provenance["selected_yaml"] == str(selected)
    assert "MODELS__SEED" not in str(config.source_provenance)


def test_explicit_yaml_environment_precedence_and_no_cross_load_leakage(tmp_path, monkeypatch):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("data:\n  source: local\n  timeframe: 1d\n")
    second.write_text("data:\n  source: bundled_demo\n  timeframe: 1h\n")
    monkeypatch.setenv("DATA__TIMEFRAME", "1h")
    monkeypatch.setenv("MODELS__SEED", "73")

    first_config = PipelineConfig.from_yaml(first)
    second_config = PipelineConfig.from_yaml(second)

    assert first_config.data.source == "local"
    assert first_config.data.timeframe.value == "1h"
    assert second_config.data.source == "bundled_demo"
    assert second_config.data.timeframe.value == "1h"
    assert first_config.models.seed == second_config.models.seed == 73


def test_resolved_config_loaders_do_not_read_default_yaml_or_environment(tmp_path, monkeypatch):
    selected = tmp_path / "selected.yaml"
    selected.write_text("data:\n  source: local\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODELS__SEED", "13")

    selected_config = PipelineConfig.from_yaml(selected)
    saved_values = selected_config.model_dump(mode="json")
    saved_values["models"]["seed"] = 99
    restored = PipelineConfig.from_artifact_config(
        saved_values,
        schema_version=config_module.PIPELINE_CONFIG_SCHEMA_VERSION,
        expected_hash=selected_config.resolved_config_hash,
    )

    assert selected_config.data.source == "local"
    assert selected_config.models.seed == 13
    assert restored.models.seed == 99
    assert restored.data.source == "local"


def test_process_environment_overrides_dotenv_and_yaml(tmp_path, monkeypatch):
    selected = tmp_path / "selected.yaml"
    dotenv = tmp_path / ".env"
    selected.write_text("models:\n  seed: 11\n")
    dotenv.write_text("MODELS__SEED=17\n")
    monkeypatch.setitem(PipelineConfig.model_config, "env_file", dotenv)
    monkeypatch.delenv("MODELS__SEED", raising=False)

    dotenv_config = PipelineConfig.from_yaml(selected)
    monkeypatch.setenv("MODELS__SEED", "19")
    environment_config = PipelineConfig.from_yaml(selected)

    assert dotenv_config.models.seed == 17
    assert environment_config.models.seed == 19


def test_direct_constructor_values_override_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS__SEED", "73")

    config = PipelineConfig(models={"seed": 94})

    assert config.models.seed == 94


@pytest.mark.parametrize(
    "reader",
    [PipelineConfig.from_yaml, IngestionConfig.from_yaml, PlatformConfig.from_yaml, CampaignMatrixConfig.from_yaml],
)
def test_explicit_config_readers_reject_missing_files(reader, tmp_path):
    with pytest.raises(FileNotFoundError):
        reader(tmp_path / "missing.yaml")


@pytest.mark.parametrize("contents", ["not: [valid", "- not-a-mapping"])
def test_pipeline_reader_reports_malformed_or_non_mapping_yaml_path(contents, tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(contents)

    with pytest.raises(ValueError, match=str(config_path)):
        PipelineConfig.from_yaml(config_path)


def test_saved_effective_config_reconstruction_ignores_new_environment(tmp_path, monkeypatch):
    selected = tmp_path / "selected.yaml"
    selected.write_text("data:\n  source: local\n  timeframe: 1d\n")
    saved = PipelineConfig.from_yaml(selected, overrides={"models": {"seed": 99}}).model_dump(mode="json")
    monkeypatch.setenv("MODELS__SEED", "13")

    restored = PipelineConfig.model_validate(saved)

    assert restored.models.seed == 99
    assert restored.data.source == "local"
    assert restored.resolved_config_hash == PipelineConfig.model_validate(saved).resolved_config_hash


@pytest.mark.parametrize(
    "section,key",
    [("data", "raw_data_dir"), ("data", "processed_data_dir"), ("validation", "require_timezone"), ("tensor", "train_cutoff_date")],
)
def test_new_configs_reject_retired_noop_settings(section, key):
    with pytest.raises(ValueError):
        PipelineConfig(**{section: {key: "unused"}})


def test_retired_flat_config_aliases_are_not_accepted():
    with pytest.raises(ValueError, match="Unknown configuration"):
        PipelineConfig(raw_data_dir="data/raw")


def test_historical_config_migration_is_explicit_and_versioned():
    old = {
        "data": {"source": "local", "raw_data_dir": "data/raw", "processed_data_dir": "data/processed"},
        "validation": {"null_tolerance": 0.001, "min_price": 0.0, "require_timezone": True},
        "tensor": {"sequence_len": 60, "batch_size": 32, "train_cutoff_date": "2025-11-20"},
    }

    restored = PipelineConfig.from_artifact_config(old, schema_version="pipeline-config.v1", expected_hash="old-hash")

    assert restored.data.source == "local"
    assert "train_cutoff_date" not in restored.model_dump()["tensor"]
    assert restored.source_provenance["migrated_legacy_fields"] == [
        "data.processed_data_dir", "data.raw_data_dir", "tensor.train_cutoff_date", "validation.require_timezone",
    ]
    assert restored.source_provenance["legacy_artifact_config_hash"] == "old-hash"
