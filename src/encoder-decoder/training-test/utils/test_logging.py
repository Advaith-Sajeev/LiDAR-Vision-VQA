from pathlib import Path

from training.utils import logging as logging_utils


class DummyStdout:
    def __init__(self):
        self.written = []
        self.flushed = False

    def write(self, s):
        self.written.append(s)

    def flush(self):
        self.flushed = True


def test_tee_creates_logfile_and_parent_dir(tmp_path):
    log_file = tmp_path / "logs" / "train.log"

    tee = logging_utils.Tee(log_file)
    try:
        # Parent dir and file should be created
        assert log_file.parent.exists()
        assert log_file.exists()
    finally:
        tee.close()


def test_tee_write_writes_to_stdout_and_file(tmp_path, monkeypatch):
    dummy = DummyStdout()
    monkeypatch.setattr(logging_utils.sys, "stdout", dummy)

    log_file = tmp_path / "out.log"
    tee = logging_utils.Tee(log_file)

    try:
        msg = "hello tee\n"
        tee.write(msg)
        tee.flush()

        # Written to original stdout
        assert "".join(dummy.written) == msg

        # Written to logfile
        assert log_file.read_text() == msg
    finally:
        tee.close()


def test_tee_close_stops_writing_to_file_but_not_stdout(tmp_path, monkeypatch):
    dummy = DummyStdout()
    monkeypatch.setattr(logging_utils.sys, "stdout", dummy)

    log_file = tmp_path / "out.log"
    tee = logging_utils.Tee(log_file)

    try:
        tee.write("before\n")
        tee.flush()
        content_before = log_file.read_text()

        tee.close()

        # After close, stdout still receives writes
        tee.write("after\n")
        tee.flush()

        # File content unchanged after close
        assert log_file.read_text() == content_before

        # Stdout did get the "after" text
        assert "".join(dummy.written).endswith("after\n")
    finally:
        # Safe if already closed
        tee.close()


def test_tee_flush_calls_stdout_flush(tmp_path, monkeypatch):
    dummy = DummyStdout()
    monkeypatch.setattr(logging_utils.sys, "stdout", dummy)

    log_file = tmp_path / "out.log"
    tee = logging_utils.Tee(log_file)

    try:
        tee.flush()
        assert dummy.flushed is True
    finally:
        tee.close()


def test_tee_close_is_idempotent(tmp_path):
    log_file = tmp_path / "logs" / "train.log"
    tee = logging_utils.Tee(log_file)

    # Calling close multiple times should not raise and should keep closed=True
    tee.close()
    tee.close()
    assert tee.closed is True
