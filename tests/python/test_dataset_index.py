"""已声明数据的一次解析、无标签输入边界和普通编号。"""

import sqlite3

import pytest
from gp_faco.dataset_index import IndexedDataset, build_index


def inputs(root):
    files = []
    for role, numbers in [
        ("train_dataset", range(8)),
        ("val_dataset", range(8, 10)),
        ("test_dataset", range(10, 12)),
    ]:
        p = root / role / "data.txt"
        p.parent.mkdir(parents=True)
        p.write_text("".join(f"0 0 {1 + i / 10} 0 0 1 output 1 2 3 1\n" for i in numbers))
        files.append({"path": str(p.relative_to(root)), "dimension": 3})
    return files, {
        "split_seed": 17,
        "development_groups_per_scale": 2,
        "validation_groups_per_scale": 3,
    }


def assignments(path):
    with sqlite3.connect(path / "instances.sqlite") as con:
        return list(
            con.execute(
                "SELECT representative_id,split,dimension FROM assignments ORDER BY "
                "representative_id"
            )
        )


def test_offsets_plain_ids_and_no_labels_in_problem(tmp_path):
    root = tmp_path / "source"
    files, policy = inputs(root)
    out = tmp_path / "index"
    assert build_index(root, files, out, policy)["records"] == 12
    with IndexedDataset(out / "instances.sqlite", root) as data:
        ids = []
        for split in ("train", "development", "validation", "test"):
            for name in data.record_ids(split, 3):
                p = data.load_instance(name)
                assert p.numeric_id > 0 and name.startswith("file")
                assert not hasattr(p, "tour") and not hasattr(p, "cost")
                assert data.load_label(name).cost > 0
                ids.append(name)
        assert len(ids) == len(set(ids)) == 12
    with pytest.raises(FileExistsError):
        build_index(root, files, out, policy)


def test_input_listing_order_and_reference_tours_do_not_choose_split(tmp_path):
    root = tmp_path / "source"
    files, policy = inputs(root)
    build_index(root, files, tmp_path / "a", policy)
    for f in files:
        p = root / f["path"]
        p.write_text(p.read_text().replace("output 1 2 3 1", "output 2 1 3 2"))
    build_index(root, list(reversed(files)), tmp_path / "b", policy)
    assert assignments(tmp_path / "a") == assignments(tmp_path / "b")
