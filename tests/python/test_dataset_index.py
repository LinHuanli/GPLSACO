import copy
import fcntl
import hashlib
import sqlite3

import pytest
from gp_faco.dataset_index import IndexedDataset, build_index


def record(number, permuted=False):
    points = [(0.0, 0.0), (1.0 + number * 0.1, 0.0), (0.3, 1.0 + number * 0.03)]
    if permuted:
        points = [points[2], points[0], points[1]]
    return " ".join(str(v) for point in points for v in point) + " output 1 2 3 1\n"


def inputs(root):
    sources = {
        "train_dataset/train.txt": [record(i) for i in range(8)],
        # 8为独立验证；2是与训练同点集的置换。不能同时进入训练。
        "val_dataset/val.txt": [record(8), record(2, permuted=True)],
        "test_dataset/test.txt": [record(9), record(3)],
    }
    metadata = []
    for path, lines in sources.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = "".join(lines).encode()
        target.write_bytes(data)
        metadata.append(
            {
                "path": path,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "dimension": 3,
                "instances": len(lines),
            }
        )
    policy = {
        "scales": [3],
        "split_seed": 17,
        "development_groups_per_scale": 2,
        "validation_groups_per_scale": 3,
        "development_anchors": [{"path": "train_dataset/train.txt", "row": 0}],
        "unknown_parentage": "unavailable",
    }
    return metadata, policy


def assignment_groups(path):
    with sqlite3.connect(path) as connection:
        return dict(connection.execute("SELECT point_set_sha256,split FROM assignments"))


def test_complete_index_group_isolation_and_offset_roundtrip(tmp_path):
    root = tmp_path / "source"
    files, policy = inputs(root)
    output = tmp_path / "index"
    summary = build_index(root, files, output, policy)
    assert summary["validated_records"] == 12
    assert summary["group_audit"]["duplicate_records_excluded"] == 2
    assert summary["group_audit"]["counts"]["3"]["groups"] == {
        "development": 2,
        "validation": 3,
        "test": 2,
        "train": 3,
    }
    with IndexedDataset(output / "instances.sqlite", root) as dataset:
        ids = []
        for split in ("train", "development", "validation", "test"):
            for record_id in dataset.record_ids(split, 3):
                instance = dataset.load_instance(record_id)
                label = dataset.load_label(record_id)
                assert instance.dimension == 3 and label.cost > 0
                assert not hasattr(instance, "tour") and not hasattr(instance, "cost")
                ids.append(record_id)
        assert len(ids) == len(set(ids)) == 10
    with pytest.raises(FileExistsError):
        build_index(root, files, output, policy)


def test_parallel_input_order_and_label_values_do_not_choose_splits(tmp_path):
    root = tmp_path / "source"
    files, policy = inputs(root)
    a, b = tmp_path / "a", tmp_path / "b"
    first = build_index(root, files, a, policy)
    second = build_index(root, list(reversed(files)), b, policy, workers=2)
    assert first["membership"] == second["membership"]
    assert first["database_sha256"] == second["database_sha256"]
    # 改变标签 tour 的编码身份，点集归属不能变化；划分排名不读取标签。
    changed_files = copy.deepcopy(files)
    for file in changed_files:
        target = root / file["path"]
        data = target.read_bytes().replace(b"output 1 2 3 1", b"output 2 1 3 2")
        target.write_bytes(data)
        file["sha256"] = hashlib.sha256(data).hexdigest()
        file["bytes"] = len(data)
    c = tmp_path / "c"
    build_index(root, changed_files, c, policy)
    assert assignment_groups(a / "instances.sqlite") == assignment_groups(c / "instances.sqlite")


def test_changed_bytes_and_malformed_labels_are_rejected(tmp_path):
    root = tmp_path / "source"
    files, policy = inputs(root)
    output = tmp_path / "index"
    build_index(root, files, output, policy)
    with IndexedDataset(output / "instances.sqlite", root) as dataset:
        record_id = dataset.record_ids("development", 3)[0]
        path = dataset.connection.execute(
            "SELECT path FROM records WHERE record_id=?", (record_id,)
        ).fetchone()[0]
        target = root / path
        data = target.read_bytes().replace(b"output 1 2 3 1", b"output 1 2 2 1")
        target.write_bytes(data)
        with pytest.raises(ValueError, match="字节"):
            dataset.load_instance(record_id)
    with pytest.raises(ValueError):
        build_index(root, files, tmp_path / "corrupted", policy)


def test_existing_writer_and_insufficient_validation_groups_fail(tmp_path):
    root = tmp_path / "source"
    files, policy = inputs(root)
    output = tmp_path / "locked"
    output.mkdir()
    with (output / "writer.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="持有"):
            build_index(root, files, output, policy)
    policy["validation_groups_per_scale"] = 100
    with pytest.raises(ValueError, match="不足"):
        build_index(root, files, tmp_path / "insufficient", policy)
    assert not (tmp_path / "insufficient/instances.sqlite").exists()
