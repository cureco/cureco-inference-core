"""異常検知モジュール（PatchCoreLite）の単体テスト。

torch が必要（`pip install cureco-inference-core[anomaly]`）。
バックボーン重みのダウンロードを避けるため、環境変数 CURECO_ANOMALY_WEIGHTS に
WideResNet50 の state_dict（.pt）を指定して実行する（未指定時はダウンロードを試みる）。
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cureco_inference_core.anomaly import PatchCoreLite  # noqa: E402

WEIGHTS = os.environ.get("CURECO_ANOMALY_WEIGHTS")


@pytest.fixture(scope="module")
def judge():
    rng = np.random.default_rng(0)
    crops = rng.integers(90, 130, size=(24, 96, 96), dtype=np.uint8)  # 一様ノイズ面=正常
    j = PatchCoreLite(coreset_size=256, pre_subsample=5000, weights_path=WEIGHTS)
    j.fit(crops)
    return j


def test_fit_and_threshold(judge):
    assert judge.bank is not None and judge.bank.shape[1] == 1536
    assert judge.color is False
    thr = judge.threshold()
    assert thr > 0


def test_scores_separate_anomaly(judge):
    rng = np.random.default_rng(1)
    normal = rng.integers(90, 130, size=(4, 96, 96), dtype=np.uint8)
    defect = normal.copy()
    defect[:, 30:60, 30:60] = 255   # 明確な異物
    s_n, maps = judge.score_crops(normal)
    s_d, _ = judge.score_crops(defect)
    assert maps.shape[0] == 4
    assert float(s_d.mean()) > float(s_n.mean()), "異常が正常より高スコアであること"


def test_tile_and_dense_calibration(judge):
    rng = np.random.default_rng(2)
    imgs = [rng.integers(90, 130, size=(300, 400), dtype=np.uint8) for _ in range(3)]
    tiles, origins = judge.tile_image(imgs[0])
    assert tiles.shape[1:] == (96, 96) and len(tiles) == len(origins)
    judge.calibrate_dense(imgs)
    assert judge.dense_scores is not None and len(judge.dense_scores) == 3
    assert judge.dense_threshold() >= float(np.max(judge.dense_scores))


def test_absorb_and_roundtrip(judge, tmp_path):
    before = judge.bank.shape[0]
    added = judge.absorb(np.full((1, 96, 96), 200, dtype=np.uint8))
    assert added > 0 and judge.bank.shape[0] == before + added

    path = tmp_path / "bank.pt"
    judge.save(path)
    loaded = PatchCoreLite(weights_path=WEIGHTS).load(path)
    assert loaded.bank.shape == judge.bank.shape
    assert loaded.color == judge.color
    assert loaded.dense_scores is not None   # calibrate_dense 済みが往復する


def test_color_bank():
    rng = np.random.default_rng(3)
    crops = rng.integers(90, 130, size=(16, 96, 96, 3), dtype=np.uint8)
    j = PatchCoreLite(coreset_size=128, pre_subsample=3000, weights_path=WEIGHTS)
    j.fit(crops)
    assert j.color is True
    s, _ = j.score_crops(crops[:2])
    assert s.shape == (2,)
