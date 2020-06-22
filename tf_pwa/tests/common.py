import tempfile
import contextlib
import os

@contextlib.contextmanager
def write_temp_file(s, filename=None):
    if filename is None:
        a = tempfile.mktemp()
    else:
        a = filename
    with open(a, "w") as f:
        f.write(s)
    yield a
    os.remove(a)
