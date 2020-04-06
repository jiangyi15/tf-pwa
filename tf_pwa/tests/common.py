import tempfile
import contextlib
import os

@contextlib.contextmanager
def write_temp_file(s):
    a = tempfile.mktemp()
    with open(a, "w") as f:
        f.write(s)
    yield a
    os.remove(a)
