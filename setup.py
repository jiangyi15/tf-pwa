from setuptools import setup, find_packages

version = {}
with open("tf_pwa/version.py") as fp:
    exec(fp.read(), version)
# later on we use: version['__version__']

with open("README.md", "r") as fh:
    long_description = fh.read()

name = "TFPWA"

setup(
    name=name,  # Replace with your own username
    version=version["__version__"],
    author="Yi Jiang",
    author_email="jiangyi15@mails.ucas.ac.cn",
    description="Partial Wave Analysis program using Tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(exclude=["benchmarks"]),
    package_data={
        # If any package contains files, include them:
        "": ["*.yml", "*.json"],
        # And include any *.json files found in the "tf_pwa" package, too:
        "tf_pwa": ["*.json"],
    },
    scripts=[
        "state_cache.sh",
    ],
    data_files=[
        "Resonances.yml.sample",
        "config.yml.sample",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "fit_pwa = fit:fit",
            "tf_pwa = tf_pwa.__main__:main"
        ],
    },
    install_requires=[
        "tensorflow>=2.0,<=2.2",
        "matplotlib",
        "sympy",
        "PyYAML",
        "opt_einsum",
    ],
    extras_require = { 
        "root": ["uproot"],
        "minuit": ["iminuit"],
        "all": ["uproot", "iminuit"]
    },
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version["__version__"]),
            'release': ('setup.py', version["__version__"]),
            'source_dir': ('setup.py', 'docs')
        }
    }
)
