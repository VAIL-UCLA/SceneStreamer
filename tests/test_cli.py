import importlib.util

import pytest


def test_cli_help_runs(capsys):
    from scenestreamer.cli import main

    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0

    out = capsys.readouterr().out
    assert "SceneStreamer paper reproduction CLI" in out


def test_preprocess_missing_scenarionet_message():
    # This repo treats scenarionet as an external git dependency.
    if importlib.util.find_spec("scenarionet") is not None:
        pytest.skip("scenarionet is installed; skip missing-dep UX test")

    from scenestreamer.cli import main

    with pytest.raises(SystemExit) as e:
        main(["preprocess", "--limit", "1"])
    assert "scenarionet" in str(e.value).lower()

