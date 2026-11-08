"""Build the ``dense_factor`` TensorFlow custom op.

The op is a fused contraction for the dense-factor PWA model::

    den[u,e] = sum_i g[off[u]+i] * B_t[e, off[u]+i]
    A[e, p]  = sum_u F[e,p,u] * C[u] / den[u,e]

It performs the forward pass and the analytic gradients with respect to
``C`` and ``g`` in two kernels, instead of a graph of small TensorFlow ops.
The forward returns ``A``; the probability is a density-matrix contraction
on top of it.

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
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_CC = _DIR / "dense_factor_op.cc"
_CU = _DIR / "dense_factor_op.cu"

#: Flags used for both the device and host compilation, on top of the ones
#: reported by ``tf.sysconfig`` (see :func:`_tf_flags`).
COMPILE_FLAGS = ("-O3",)


CACHE_ENV = "TF_PWA_CACHE_DIR"


def _writable(path):
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_test"
    probe.write_bytes(b"")
    probe.unlink()
    return path


def _cache_dir():
    """Directory holding the op library.

    Resolution order:

    1. ``$TF_PWA_CACHE_DIR``, always a directory.  A library already in
       it is reused, which is how a prebuilt ``.so`` is deployed.
    2. ``$XDG_CACHE_HOME/tf_pwa/dense_factor_op`` (default
       ``~/.cache/tf_pwa/dense_factor_op``).
    3. The package directory, if neither of the above is writable.

    The package directory is read-only under a normal ``site-packages``
    install, which is why the user cache is preferred.
    """
    explicit = os.environ.get(CACHE_ENV)
    if explicit:
        path = Path(explicit)
        if path.exists() and not path.is_dir():
            raise RuntimeError(f"{CACHE_ENV}={explicit!r} is not a directory")
        try:
            return _writable(path)
        except OSError as exc:
            raise RuntimeError(
                f"{CACHE_ENV}={explicit!r} is not writable: {exc}"
            )
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    try:
        return _writable(Path(base) / "tf_pwa" / "dense_factor_op")
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


def _tf_flags():
    """Compile and link flags reported by the installed TensorFlow.

    Taken from ``tf.sysconfig`` instead of being hard-coded so that the
    build follows the wheel it links against.  The libstdc++ ABI
    (``_GLIBCXX_USE_CXX11_ABI``), the C++ standard, Eigen's alignment
    define and the extra include roots are all release-specific.  Pinning
    ``-D_GLIBCXX_USE_CXX11_ABI=1`` compiles cleanly against a wheel built
    with the pre-C++11 ABI and then fails to link; that is the failure an
    older GPU stack (sm_70 with TensorFlow 2.9 and CUDA 11, say) would hit.

    Returns ``(compile_flags, link_flags)``.
    """
    import tensorflow as tf

    compile_flags = [
        str(f).replace("--std=", "-std=")
        for f in tf.sysconfig.get_compile_flags()
    ]
    link_flags = [str(f) for f in tf.sysconfig.get_link_flags()]
    # The wheel states its own standard; only pick a default when it does not.
    if not any(f.startswith("-std=") for f in compile_flags):
        compile_flags.append("-std=c++17")
    return compile_flags, link_flags


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
    """Digest of the kernel sources.

    Compile flags, the host compiler and the CUDA toolkit are not part of
    it: they are environment, not identity, and including them would stop
    a binary built on another machine from matching.  After changing any
    of them, rebuild with ``force=True``.
    """
    h = hashlib.sha256()
    for p in (_CC, _CU):
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def build(force=False):
    """Return the path of the compiled op, building it if needed.

    A library named for the current source digest, TensorFlow version and
    arch is reused instead of building, so a prebuilt ``.so`` is deployed
    by copying it into ``$TF_PWA_CACHE_DIR`` under that exact name.  Any
    other file there is ignored.

    The digest covers only the kernel sources.  Changing a compile flag,
    the host compiler or the CUDA toolkit therefore does not invalidate
    the cache; use ``force=True`` or ``--force`` after such a change.
    """
    cache = _cache_dir()

    import tensorflow as tf

    tf_inc = Path(tf.sysconfig.get_include())
    tf_lib = Path(tf.sysconfig.get_lib())

    arch = os.environ.get("DENSE_OP_ARCH") or _arch()
    digest = _source_hash()
    tag = f"{digest}-tf{tf.__version__}-{arch}"
    so = cache / f"_dense_factor_op-{tag}.so"
    if not force and so.exists():
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

    tf_cc, tf_ld = _tf_flags()
    inc = list(
        dict.fromkeys(
            [
                f"-I{cuda_inc}",
                f"-I{tf_inc}",
                f"-I{absl_inc}",
                *tf_cc,
                *COMPILE_FLAGS,
                "-w",
            ]
        )
    )
    link = list(tf_ld) or [f"-L{tf_lib}", "-l:libtensorflow_framework.so.2"]
    if not any(f.startswith("-Wl,-rpath") for f in link):
        link.append(f"-Wl,-rpath,{tf_lib}")
    # The device translation unit uses `if constexpr` and includes only CUDA
    # headers, so its dialect is pinned instead of inherited from the wheel:
    # nvcc otherwise gets the wheel's standard (c++14 on older releases) and
    # takes the REG branch as a non-standard extension.
    cu_inc = [f for f in inc if not f.startswith("-std=")] + ["-std=c++17"]
    # Intermediates carry the pid: two processes may build the same digest at
    # once, and the objects are not reused anyway (a cached digest is found by
    # its ``.so``), so they are deleted once the library is in place.
    stem = f"_dense_factor_op-{tag}.{os.getpid()}"
    cc_obj = cache / f"{stem}.cc.o"
    cu_obj = cache / f"{stem}.cu.o"
    tmp_so = cache / f"{stem}.tmp.so"

    cmds = [
        [
            shutil.which("nvcc") or "nvcc",
            "-c",
            str(_CU),
            "-o",
            str(cu_obj),
            *cu_inc,
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
            str(tmp_so),
            *link,
        ],
    ]
    try:
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
        # Publish atomically: a build that dies mid-link must not leave a
        # truncated library that the existence check would then accept.
        os.replace(tmp_so, so)
    finally:
        for path in (cc_obj, cu_obj, tmp_so):
            try:
                path.unlink()
            except OSError:
                pass
    return str(so)


def _prune_cache(cache, so_path):
    """Drop stale libraries and the leftovers of dead builds.

    Libraries for a different ``-tf<version>-sm<XX>`` suffix are kept, so a
    shared cache can hold several targets, but objects and half-written
    libraries are transient for every target and always removed.  Fresh
    transients are left alone: they belong to a build running right now.
    Files that are not ``_dense_factor_op-*`` are never touched.
    """
    name = Path(so_path).name
    match = re.match(r"_dense_factor_op-(.*)(-tf.+-sm[\d_]+)\.so$", name)
    if match is None:
        return []
    keep = name[: -len(".so")]
    suffix = match.group(2)  # -tf2.21.0-sm_120

    removed = []
    now = time.time()
    for path in sorted(Path(cache).glob("_dense_factor_op-*")):
        # Objects and half-written libraries are transient: they are never
        # reused and are only here because a build died before cleaning up.
        if path.suffix == ".o" or ".tmp." in path.name:
            try:
                if now - path.stat().st_mtime < 300:
                    continue  # a build is writing it
            except OSError:
                continue
        elif suffix not in path.name:
            continue  # another TF version or arch: keep
        elif path.name.startswith(keep):
            continue  # the build just resolved
        try:
            path.unlink()
            removed.append(path.name)
        except OSError:
            pass
    return removed


def main(argv=None):
    """Build the op from the command line, e.g. for CI artefacts.

    ``python -m tf_pwa.experimental.dense_factor_op.build`` prints the
    path of the library, building it if the cache has no match.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m tf_pwa.experimental.dense_factor_op.build",
        description="Build the dense-factor custom op and print its path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild even if a cached library matches",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        help=f"build into DIR (same as {CACHE_ENV})",
    )
    parser.add_argument(
        "--arch",
        default=None,
        metavar="sm_XX",
        help="target compute capability (same as DENSE_OP_ARCH)",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="delete other _dense_factor_op-* files from the cache dir",
    )
    args = parser.parse_args(argv)

    if args.cache_dir:
        os.environ[CACHE_ENV] = args.cache_dir
    if args.arch:
        os.environ["DENSE_OP_ARCH"] = args.arch
    so = build(force=args.force)
    print(so)
    if args.prune:
        for name in _prune_cache(_cache_dir(), so):
            print("pruned " + name, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
