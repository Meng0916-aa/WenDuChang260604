"""
Tests for scripts/02c_batch_convert_tracks.py and the shared parse core.

Uses tiny SYNTHETIC .xtherm files (56-byte header + small uint16 payload) and a
synthetic master list in a temp dir — no real data is touched. Covers the 16
behaviours required for the per-track batch converter.
"""

import os
import sys
import json
import importlib.util

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from conversion.xtherm_binary import build_numpy_dtype, read_xtherm_frame, XthermSizeError

HEADER = 56
W = 3
H = 2
SCALE = 0.1


def _load_02c():
    path = os.path.join(_ROOT, "scripts", "02c_batch_convert_tracks.py")
    spec = importlib.util.spec_from_file_location("s02c", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load_02c()


def _fmt():
    nd = build_numpy_dtype("uint16", "little")
    return {
        "width": W, "height": H, "header_bytes": HEADER,
        "dtype": "uint16", "endian": "little", "scale_factor": SCALE,
        "np_dtype": nd, "expected_size": HEADER + W * H * 2,
        "valid_max": 3000.0, "zero_ratio_note_threshold": 0.05,
    }


def _write_xtherm(path, base, header=HEADER):
    payload = np.arange(base, base + W * H, dtype="<u2").reshape(H, W)
    with open(path, "wb") as f:
        f.write(b"\x00" * header)
        f.write(payload.tobytes())


def _make_track(folder, n, start=1, with_session=True, base_step=100):
    os.makedirs(folder, exist_ok=True)
    for k in range(n):
        _write_xtherm(os.path.join(folder, f"{start + k:04d}.xtherm"),
                      base=(k + 1) * base_step)
    if with_session:
        with open(os.path.join(folder, "session.xml"), "w", encoding="utf-8") as f:
            f.write("<?xml version='1.0'?><session/>")
    return folder


def _row(sample_id, folder, n, speed=600, cond="R1", track="T1", order=1):
    return {
        "condition_id": cond, "track_id": track, "sample_id": sample_id,
        "design_role": "box_behnken_edge", "laser_power_W": "300",
        "scan_speed_mm_min": str(speed), "magnetic_field_mT": "60",
        "powder_feed_g_min": "40", "travel_distance_mm": "30",
        "raw_folder": folder.replace("\\", "/"),
        "session_xml": os.path.join(folder, "session.xml").replace("\\", "/"),
        "xtherm_count": str(n), "track_order": str(order),
        "processing_status": "raw_only", "notes": "",
    }


# 1 -------------------------------------------------------------------------
def test_natural_sort_2_before_10(tmp_path):
    from conversion.xtherm_binary import natural_sort_key, list_xtherm_files
    assert sorted(["10.xtherm", "2.xtherm", "1.xtherm"], key=natural_sort_key) == \
        ["1.xtherm", "2.xtherm", "10.xtherm"]
    d = tmp_path / "trk"
    d.mkdir()
    for nm, base in (("10.xtherm", 999), ("2.xtherm", 200), ("1.xtherm", 100)):
        _write_xtherm(d / nm, base)
    files = list_xtherm_files(str(d), recursive=False)
    assert [os.path.basename(f) for f in files] == ["1.xtherm", "2.xtherm", "10.xtherm"]


# 2 -------------------------------------------------------------------------
def test_56_byte_header_parse(tmp_path):
    p = tmp_path / "0001.xtherm"
    _write_xtherm(p, base=100)
    nd = build_numpy_dtype("uint16", "little")
    frame = read_xtherm_frame(str(p), W, H, HEADER, nd, SCALE)
    assert frame.shape == (H, W)
    np.testing.assert_allclose(frame, np.arange(100, 106).reshape(H, W) * 0.1, atol=1e-6)


# 3 -------------------------------------------------------------------------
def test_file_size_validation(tmp_path):
    folder = _make_track(str(tmp_path / "R1" / "T1"), n=2)
    bad = os.path.join(folder, "0003.xtherm")
    with open(bad, "wb") as f:
        f.write(b"\x00" * (HEADER + 2))   # wrong size
    row = _row("R1_T1", folder, n=3)
    v = m.validate_track(row, _fmt())
    assert v["ok"] is False
    assert any("bad size" in r for r in v["reasons"])
    # core raises on the truncated file directly
    nd = build_numpy_dtype("uint16", "little")
    with pytest.raises(XthermSizeError):
        read_xtherm_frame(bad, W, H, HEADER, nd, SCALE)


# 4 -------------------------------------------------------------------------
def test_uint16_to_float32_scale(tmp_path):
    p = tmp_path / "0001.xtherm"
    _write_xtherm(p, base=250)
    nd = build_numpy_dtype("uint16", "little")
    frame = read_xtherm_frame(str(p), W, H, HEADER, nd, SCALE)
    assert frame.dtype == np.float32
    assert abs(float(frame[0, 0]) - 25.0) < 1e-6


# 5 -------------------------------------------------------------------------
def test_single_output_naming(tmp_path):
    folder = _make_track(str(tmp_path / "R1" / "T1"), n=3)
    out_root = str(tmp_path / "out")
    meta = m.convert_track(_row("R1_T1", folder, 3), _fmt(), out_root)
    assert meta["conversion_status"] == "converted"
    assert os.path.isfile(os.path.join(out_root, "matrix", "R1_T1.npy"))
    assert os.path.isfile(os.path.join(out_root, "matrix_meta", "R1_T1.json"))


# 6 -------------------------------------------------------------------------
def test_never_writes_dataset_npy(tmp_path):
    folder = _make_track(str(tmp_path / "R1" / "T1"), n=3)
    out_root = str(tmp_path / "out")
    meta = m.convert_track(_row("R1_T1", folder, 3), _fmt(), out_root)
    assert os.path.basename(meta["output_file"]) == "R1_T1.npy"
    assert not os.path.exists(os.path.join(out_root, "matrix", "dataset.npy"))
    # a 'dataset' sample_id is refused outright
    bad = m.convert_track(_row("dataset", folder, 3), _fmt(), out_root)
    assert bad["conversion_status"] == "fail"


# 7 -------------------------------------------------------------------------
def test_count_mismatch_fails(tmp_path):
    folder = _make_track(str(tmp_path / "R1" / "T1"), n=3)
    row = _row("R1_T1", folder, n=5)   # master claims 5, only 3 exist
    v = m.validate_track(row, _fmt())
    assert v["ok"] is False
    assert any("count mismatch" in r for r in v["reasons"])
    meta = m.convert_track(row, _fmt(), str(tmp_path / "out"))
    assert meta["conversion_status"] == "fail"


# 8 -------------------------------------------------------------------------
def test_missing_session_xml_fails(tmp_path):
    folder = _make_track(str(tmp_path / "R1" / "T1"), n=3, with_session=False)
    v = m.validate_track(_row("R1_T1", folder, 3), _fmt())
    assert v["ok"] is False
    assert any("session.xml" in r for r in v["reasons"])


# 9 -------------------------------------------------------------------------
def test_existing_output_skips(tmp_path):
    folder = _make_track(str(tmp_path / "R1" / "T1"), n=3)
    out_root = str(tmp_path / "out")
    first = m.convert_track(_row("R1_T1", folder, 3), _fmt(), out_root)
    assert first["conversion_status"] == "converted"
    again = m.convert_track(_row("R1_T1", folder, 3), _fmt(), out_root)
    assert again["conversion_status"] == "skipped"


# 10 ------------------------------------------------------------------------
def test_overwrite_reconverts(tmp_path):
    folder = _make_track(str(tmp_path / "R1" / "T1"), n=3)
    out_root = str(tmp_path / "out")
    m.convert_track(_row("R1_T1", folder, 3), _fmt(), out_root)
    again = m.convert_track(_row("R1_T1", folder, 3), _fmt(), out_root, overwrite=True)
    assert again["conversion_status"] == "converted"


# 11 ------------------------------------------------------------------------
def test_tmp_failure_leaves_no_official_output(tmp_path, monkeypatch):
    folder = _make_track(str(tmp_path / "R1" / "T1"), n=3)
    out_root = str(tmp_path / "out")

    def boom(*a, **k):
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr(m.np, "save", boom)
    meta = m.convert_track(_row("R1_T1", folder, 3), _fmt(), out_root)
    assert meta["conversion_status"] == "fail"
    assert not os.path.exists(os.path.join(out_root, "matrix", "R1_T1.npy"))
    assert not os.path.exists(os.path.join(out_root, "matrix", "R1_T1.npy.tmp"))


# 12 ------------------------------------------------------------------------
def test_dry_run_writes_nothing(tmp_path):
    folder = _make_track(str(tmp_path / "R1" / "T1"), n=3)
    out_root = str(tmp_path / "out")
    total = m.dry_run([_row("R1_T1", folder, 3)], _fmt(), out_root)
    assert total == 3 * H * W * 4
    assert not os.path.exists(os.path.join(out_root, "matrix"))


# 13 ------------------------------------------------------------------------
def test_sample_id_scopes_to_one(tmp_path):
    f1 = _make_track(str(tmp_path / "R1" / "T1"), n=3)
    f2 = _make_track(str(tmp_path / "R2" / "T1"), n=3)
    rows = [_row("R1_T1", f1, 3, cond="R1"), _row("R2_T1", f2, 3, cond="R2")]
    args = type("A", (), {"all": False, "select_representative": False,
                          "sample_id": "R2_T1", "sample_ids": None})()
    targets = m.enumerate_targets(rows, args)
    assert [t["sample_id"] for t in targets] == ["R2_T1"]


# 14 ------------------------------------------------------------------------
def test_all_enumerates_every_row_without_converting(tmp_path):
    rows = []
    for i in range(1, 6):
        f = _make_track(str(tmp_path / f"R{i}" / "T1"), n=2)
        rows.append(_row(f"R{i}_T1", f, 2, cond=f"R{i}"))
    args = type("A", (), {"all": True, "select_representative": False,
                          "sample_id": None, "sample_ids": None})()
    targets = m.enumerate_targets(rows, args)
    assert len(targets) == len(rows)
    out_root = str(tmp_path / "out")
    m.dry_run(targets, _fmt(), out_root)        # listing only
    assert not os.path.exists(os.path.join(out_root, "matrix"))


def test_real_master_all_sees_57_if_present():
    """If the real local master exists, --all must enumerate exactly 57 tracks."""
    csv_path = os.path.join(_ROOT, "data", "metadata", "experiment_master.csv")
    if not os.path.isfile(csv_path):
        pytest.skip("real experiment_master.csv not present")
    rows = m.read_master(csv_path)
    args = type("A", (), {"all": True, "select_representative": False,
                          "sample_id": None, "sample_ids": None})()
    assert len(m.enumerate_targets(rows, args)) == 57


# 15 ------------------------------------------------------------------------
def test_excludes_dataset_directory(tmp_path):
    with pytest.raises(ValueError):
        m.assert_not_dataset("data/raw_xtherm/dataset/R1/T1")
    folder = _make_track(str(tmp_path / "dataset" / "R1" / "T1"), n=2)
    v = m.validate_track(_row("R1_T1", folder, 2), _fmt())
    assert v["ok"] is False
    assert any("dataset" in r for r in v["reasons"])


# 16 ------------------------------------------------------------------------
def test_reloaded_output_shape_and_dtype(tmp_path):
    folder = _make_track(str(tmp_path / "R1" / "T1"), n=4)
    out_root = str(tmp_path / "out")
    meta = m.convert_track(_row("R1_T1", folder, 4), _fmt(), out_root)
    arr = np.load(meta["output_file"])
    assert arr.shape == (4, H, W)
    assert arr.dtype == np.float32
    # values follow natural frame order: frame k base=(k+1)*100 -> [k+1 ... ]*0.1
    assert abs(float(arr[0, 0, 0]) - 10.0) < 1e-6
    assert abs(float(arr[3, 0, 0]) - 40.0) < 1e-6


# extra: representative selection determinism ------------------------------
def test_select_representative_prefers_t1_closest_to_median(tmp_path):
    rows = []
    # speed 400: counts 230,240,240,255 -> median 240 -> prefer a T1 at 240
    specs = [("R1", "T1", 255), ("R2", "T1", 240), ("R2", "T2", 240), ("R9", "T1", 230)]
    for cond, trk, n in specs:
        rows.append(_row(f"{cond}_{trk}", str(tmp_path / cond / trk), n,
                         speed=400, cond=cond, track=trk,
                         order=int(trk[1])))
    chosen = m.select_representative(rows)
    assert len(chosen) == 1
    assert chosen[0]["speed"] == 400
    assert chosen[0]["sample_id"] == "R2_T1"   # 240 == median, T1, first by order


if __name__ == "__main__":
    import tempfile
    print("run via: pytest tests/test_batch_convert_tracks.py -q")
