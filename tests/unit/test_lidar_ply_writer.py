import numpy as np

from infinigen.lidar.lidar_engine import save_ply


def test_ascii_ply_int_columns_are_written_as_ints(tmp_path):
    """Ensure ASCII PLY writer does not emit integer fields with decimal suffixes."""
    out = tmp_path / "pc.ply"
    data = {
        "points": np.array([[1.0, -2.0, 3.5]], dtype=np.float32),
        "intensity": np.array([7], dtype=np.uint8),
        "ring": np.array([9], dtype=np.uint16),
        "azimuth": np.array([0.1], dtype=np.float32),
        "elevation": np.array([-0.2], dtype=np.float32),
        "return_id": np.array([2], dtype=np.uint8),
        "num_returns": np.array([1], dtype=np.uint8),
        "mat_class": np.array([5], dtype=np.uint8),
    }

    save_ply(out, data, binary=False)
    lines = out.read_text().strip().splitlines()
    end_header = lines.index("end_header")
    row = lines[end_header + 1].split()
    assert len(row) == 10  # base 9 + mat_class

    def is_int_token(tok):
        return tok.lstrip("-").isdigit()

    assert is_int_token(row[3])  # intensity
    assert is_int_token(row[4])  # ring
    assert is_int_token(row[7])  # return_id
    assert is_int_token(row[8])  # num_returns
    assert is_int_token(row[9])  # mat_class (optional integer)
