"""Build the ``dense_factor`` TensorFlow custom op.

The op is a fused contraction for the dense-factor PWA model::

    den[u,e] = sum_i g[off[u]+i] * B_t[e, off[u]+i]
    A[e]     = sum_u F[e,u] * C[u] / den[u,e]
    Q        = sum_e w[e] * |A[e]|**2

It performs the forward pass and the analytic gradients with respect to
``C`` and ``g`` in two kernels, instead of a graph of small TensorFlow ops.

The build is done once and cached next to this file.  It needs:

* the TensorFlow headers/libraries from ``tf.sysconfig`` (they must be
  present in the installed wheel),
* a CUDA toolkit whose version is compatible with the TensorFlow build,
* the abseil/protobuf headers that TensorFlow's own headers include
  (conda environments ship them in ``$CONDA_PREFIX/include``; the wheel
  does not).

If any of those are missing the build fails with a message naming the
missing piece; nothing else in ``tf_pwa`` depends on this module.
"""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_CC = _DIR / "dense_factor_op.cc"
_CU = _DIR / "dense_factor_op.cu"


def _cache_dir():
    """Writable directory for the compiled library.

    The package directory is read-only under a normal ``site-packages``
    install, so prefer the user cache and only fall back to the package
    directory when no cache can be created.
    """
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    cand = Path(base) / "tf_pwa" / "dense_factor_op"
    try:
        cand.mkdir(parents=True, exist_ok=True)
        probe = cand / ".write_test"
        probe.write_bytes(b"")
        probe.unlink()
        return cand
    except OSError:
        return _DIR


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def _first_dir(*candidates, marker):
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if (p / marker).exists():
            return p
    return None


def _cuda_include():
    """Locate a CUDA toolkit include dir that has the full CRT headers."""
    nvcc = shutil.which("nvcc")
    cands = []
    if os.environ.get("CUDA_HOME"):
        cands.append(Path(os.environ["CUDA_HOME"]) / "include")
    if os.environ.get("CUDA_PATH"):
        cands.append(Path(os.environ["CUDA_PATH"]) / "include")
    if nvcc:
        cands.append(Path(nvcc).resolve().parent.parent / "include")
    if os.environ.get("CONDA_PREFIX"):
        cands.append(Path(os.environ["CONDA_PREFIX"]) / "include")
        cands += sorted(
            (Path(os.environ["CONDA_PREFIX"]) / "targets").glob("*/include")
        )
    cands += [Path("/usr/local/cuda/include")]
    return _first_dir(*cands, marker="crt/host_config.h")


def _absl_include():
    """Locate the abseil headers TensorFlow's headers include."""
    cands = []
    if os.environ.get("CONDA_PREFIX"):
        cands.append(Path(os.environ["CONDA_PREFIX"]) / "include")
    cands += [Path("/usr/include"), Path("/usr/local/include")]
    return _first_dir(*cands, marker="absl/base/config.h")


def _arch():
    """Compute capability of the visible GPU, e.g. ``sm_120``."""
    r = _run(
        [
            "nvidia-smi",
            "--query-gpu=compute_cap",
            "--format=csv,noheader",
        ]
    )
    if r.returncode == 0 and r.stdout.strip():
        cc = r.stdout.strip().splitlines()[0].strip().replace(".", "")
        return "sm_" + cc
    raise RuntimeError(
        "cannot detect the GPU compute capability (nvidia-smi failed); "
        "set DENSE_OP_ARCH=sm_XX"
    )


def _source_hash():
    h = hashlib.sha256()
    for p in (_CC, _CU):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build(force=False):
    """Compile the op if needed and return the path of the ``.so``."""
    import tensorflow as tf

    tf_inc = Path(tf.sysconfig.get_include())
    tf_lib = Path(tf.sysconfig.get_lib())

    cache = _cache_dir()
    digest = _source_hash()
    tag = f"{digest}-tf{tf.__version__}-{_arch()}"
    so = cache / f"_dense_factor_op-{tag}.so"
    marker = cache / f"_dense_factor_op-{tag}.hash"
    if not force and so.exists() and marker.exists():
        if marker.read_text().strip() == digest:
            return str(so)

    cuda_inc = _cuda_include()
    if cuda_inc is None:
        raise RuntimeError(
            "CUDA toolkit headers (crt/host_config.h) not found; set "
            "CUDA_HOME or install the CUDA toolkit"
        )
    absl_inc = _absl_include()
    if absl_inc is None:
        raise RuntimeError(
            "abseil headers not found; TensorFlow's own headers need them. "
            "Install libabseil / protobuf development headers "
            "(conda: `conda install -c conda-forge libabseil`)"
        )

    arch = os.environ.get("DENSE_OP_ARCH") or _arch()
    abi = "-D_GLIBCXX_USE_CXX11_ABI=1"
    inc = [
        f"-I{cuda_inc}",
        f"-I{tf_inc}",
        f"-I{absl_inc}",
        abi,
        "-std=c++17",
        "-O3",
        "-w",
    ]
    cc_obj = cache / f"_dense_factor_op-{tag}.cc.o"
    cu_obj = cache / f"_dense_factor_op-{tag}.cu.o"

    cmds = [
        [
            shutil.which("nvcc") or "nvcc",
            "-c",
            str(_CU),
            "-o",
            str(cu_obj),
            *inc,
            f"-arch={arch}",
            "--expt-relaxed-constexpr",
            "-Xcompiler",
            "-fPIC",
        ],
        [
            os.environ.get("CXX", "g++"),
            "-c",
            str(_CC),
            "-o",
            str(cc_obj),
            *inc,
            "-fPIC",
        ],
        [
            os.environ.get("CXX", "g++"),
            "-shared",
            str(cc_obj),
            str(cu_obj),
            "-o",
            str(so),
            f"-L{tf_lib}",
            "-l:libtensorflow_framework.so.2",
            f"-Wl,-rpath,{tf_lib}",
        ],
    ]
    for cmd in cmds:
        r = _run(cmd)
        if r.returncode != 0:
            raise RuntimeError(
                "dense_factor_op build failed:\n"
                + " ".join(map(str, cmd))
                + "\n"
                + r.stdout
                + r.stderr
            )
    marker.write_text(digest)
    return str(so)
