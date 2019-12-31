from setuptools import setup, find_packages

version = {}
with open("tf_pwa/version.py") as fp:
    exec(fp.read(), version)
# later on we use: version['__version__']

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="TFPWA", # Replace with your own username
    version=version["__version__"],
    author="Example Author",
    author_email="author@example.com",
    description="Partial Wave Analysis program using Tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    package_data={
        # If any package contains files, include them:
        "": ["*.yml","*.json"],
        # And include any *.json files found in the "tf_pwa" package, too:
        "tf_pwa": ["*.json"],
    },
    scripts=[
      "fit_scipy.py",
      "fit.py",
      "plot_amp.py",
      "fitfractions.py",
      "fitfractions_toy.py",
      "state_cache.sh",
      "for_fit.sh",
    ],
    data_files = [
        "Resonances.yml",
        "Resonances.json",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        #"License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    entry_points={
      "console_scripts": [
        "fit_scipy = fit_scipy:fit",
      ]
    },
)
