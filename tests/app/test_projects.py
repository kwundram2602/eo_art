import pytest
from omegaconf import OmegaConf

from eo_art.app.projects import (
    MARKER,
    adopt_folder,
    candidate_folders,
    copy_project,
    create_project,
    list_projects,
    validate_name,
)
from eo_art.app.templates import ACQUIRE_FILENAME, RENDER_FILENAME


def test_create_project_builds_layout_and_seeds_configs(tmp_path):
    project = create_project(tmp_path, "alps")

    assert (project.root / MARKER).is_file()
    assert project.aoi_dir.is_dir()
    assert project.logs_dir.is_dir()
    assert [p.name for p in project.config_files()] == [
        ACQUIRE_FILENAME,
        RENDER_FILENAME,
    ]
    render = OmegaConf.load(project.configs_dir / RENDER_FILENAME)
    assert render.run.name == "alps"


def test_create_project_refuses_existing_folder(tmp_path):
    (tmp_path / "alps").mkdir()
    with pytest.raises(FileExistsError):
        create_project(tmp_path, "alps")


@pytest.mark.parametrize("bad", ["", "a b", "../x", "x/y"])
def test_validate_name_rejects_unsafe_names(bad):
    with pytest.raises(ValueError):
        validate_name(bad)


def test_list_projects_only_returns_marked_folders(tmp_path):
    create_project(tmp_path, "b")
    create_project(tmp_path, "a")
    (tmp_path / "plain").mkdir()

    assert [p.name for p in list_projects(tmp_path)] == ["a", "b"]
    assert [p.name for p in candidate_folders(tmp_path)] == ["plain"]


def test_adopt_folder_keeps_existing_configs(tmp_path):
    configs = tmp_path / "old" / "configs"
    configs.mkdir(parents=True)
    (configs / RENDER_FILENAME).write_text("run: {name: mine}\n")

    project = adopt_folder(tmp_path, "old")

    assert (project.root / MARKER).is_file()
    assert (configs / RENDER_FILENAME).read_text() == "run: {name: mine}\n"
    assert (configs / ACQUIRE_FILENAME).is_file()


def test_missing_projects_folder_is_created_on_first_project(tmp_path):
    projects = tmp_path / "projects"
    assert list_projects(projects) == []
    assert candidate_folders(projects) == []

    project = create_project(projects, "alps")

    assert project.root == projects / "alps"
    assert [p.name for p in list_projects(projects)] == ["alps"]


def test_copy_project_copies_top_level_configs_and_aois_only(tmp_path):
    source = create_project(tmp_path, "alps")
    (source.configs_dir / "acquire.yaml").write_text("aoi_path: aoi/valley.gpkg\n")
    (source.configs_dir / "look.yml").write_text("render: {}\n")
    (source.configs_dir / "looks").mkdir()
    (source.configs_dir / "looks" / "nested.yaml").write_text("x: 1\n")
    (source.aoi_dir / "valley.gpkg").write_bytes(b"gpkg")
    (source.out_dir / "dtm.tif").write_bytes(b"dtm")
    (source.logs_dir / "run.log").write_text("log")

    copy = copy_project(tmp_path, source, "alps_v2")

    assert copy.root == tmp_path / "alps_v2"
    assert (copy.root / MARKER).is_file()
    assert copy.logs_dir.is_dir()
    assert [p.name for p in copy.config_files()] == [
        "acquire.yaml",
        "look.yml",
        RENDER_FILENAME,
    ]
    # Copied verbatim: relative aoi paths keep pointing into the new project.
    assert (copy.configs_dir / "acquire.yaml").read_text() == (
        "aoi_path: aoi/valley.gpkg\n"
    )
    assert not (copy.configs_dir / "looks").exists()
    assert [p.name for p in copy.aoi_files()] == ["valley.gpkg"]
    assert not (copy.out_dir / "dtm.tif").exists()
    assert list(copy.logs_dir.iterdir()) == []
    assert copy in list_projects(tmp_path)


def test_copy_project_refuses_existing_target_and_bad_names(tmp_path):
    source = create_project(tmp_path, "alps")
    create_project(tmp_path, "taken")

    with pytest.raises(FileExistsError):
        copy_project(tmp_path, source, "taken")
    with pytest.raises(ValueError):
        copy_project(tmp_path, source, "bad name")
