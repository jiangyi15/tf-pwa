Tensorflow and Cudatoolkit Version
----------------------------------
1. **Why are there two separate conda requirements file?**
    - `requirements-min.txt` limits the tensorflow version up to `2.2`. Beyond this version, `conda` will install the wrong dependency versions, in particular `cudatoolkit` versions and sometimes `python3`.
    - `tensorflow_2_6_requirements.txt` manually selects the correct `python` and `cudatoolkit` versions to match the `tensorflow-2.6.0` build on `conda-forge`.

2. **Should I use the latest** `tensorflow` **version?**
    - We **highly recommend** Ampere card users (RTX 30 series for example), to install their `conda` environments with `tensorflow_2_6_requirements.txt` which uses `cudatoolkit` version **11.2**.

3. **Why should Ampere use** `cudatoolkit` **version > 11.0?**
    - To avoid *a few minutes* of overhead due to JIT compilation.
    - `cudatoolkit` version < **11.0** does not have pre-compiled CUDA binaries for Ampere architecture. So older `cudatoolkit` versions have to JIT compile the PTX code everytime `tensorflow` uses the GPU hence the overhead.
    - See this `explanation <https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/>`_ about old CUDA versions and JIT compile.

4. **Will you update the** `tensorflow_2_X_requirements.txt` **file regularly to the latest available version on `conda`?**
    - We do not guarantee any regular updates on `tensorflow_2_X_requirements.txt`.
    - We will update this should particular build become unavailable on `conda` **or** a new release of GPUs require a `tensorflow` and `cudatoolkit` update. Please notify us if this is the case.
