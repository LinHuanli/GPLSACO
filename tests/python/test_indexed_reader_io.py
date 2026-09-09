"""记录读取使用偏移与普通标签表，无需文件或路线摘要列。"""

import sqlite3

import pytest
from gp_faco.dataset_index import IndexedDataset


def fixture_database(tmp_path):
    line = "0 0 1 0 1 1 0 1 output 1 2 3 4 1\n"
    (tmp_path / "instance.txt").write_text(line)
    database = tmp_path / "instances.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            "CREATE TABLE metadata(key TEXT,value TEXT);"
            "INSERT INTO metadata VALUES('status','complete');"
            "CREATE TABLE records(record_id TEXT,path TEXT,byte_offset INT,byte_length INT);"
            "CREATE TABLE labels(record_id TEXT,cost REAL);"
            "INSERT INTO labels VALUES('instance-1',4.0);"
        )
        connection.execute(
            "INSERT INTO records VALUES(?,?,?,?)",
            ("instance-1", "instance.txt", 0, len(line.encode())),
        )
    return database


def test_instance_and_label_read_from_plain_index(tmp_path):
    with IndexedDataset(fixture_database(tmp_path), tmp_path) as source:

        def forbid_labels(action, table, *_args):
            return (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_READ and table == "labels"
                else sqlite3.SQLITE_OK
            )

        source.connection.set_authorizer(forbid_labels)
        instance = source.load_instance("instance-1")
        assert instance.dimension == 4
        assert not hasattr(instance, "tour")
        source.connection.set_authorizer(None)
        label = source.load_label("instance-1")
        assert label.tour == (0, 1, 2, 3) and label.cost == 4


def test_short_record_fails_without_reading_another_instance(tmp_path):
    database = fixture_database(tmp_path)
    (tmp_path / "instance.txt").write_text("0 0")
    with IndexedDataset(database, tmp_path) as source, pytest.raises(ValueError, match="截断"):
        source.load_instance("instance-1")
