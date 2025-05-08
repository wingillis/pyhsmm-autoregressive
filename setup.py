import setuptools
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
import shutil
import requests
import tarfile
from pathlib import Path

def download_eigen():
    eigenpath = Path('deps')
    eigenpath.mkdir(parents=True, exist_ok=True)
    eigenurl = 'https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz'
    eigentarpath = eigenpath / 'Eigen.tar.gz'
    if not eigentarpath.exists():
        print('Downloading Eigen...')
        r = requests.get(eigenurl)
        with open(eigentarpath, 'wb') as f:
            f.write(r.content)
    with tarfile.open(eigentarpath, 'r') as tar:
        tar.extractall('deps')
    if (eigenpath / "Eigen").exists():
        shutil.rmtree(eigenpath / "Eigen")
    shutil.move(eigenpath / 'eigen-3.3.7' / "Eigen", eigenpath / "Eigen")
    print('...done!')

download_eigen()

extensions = []

for file in Path("autoregressive").glob("**/*.pyx"):
    extensions.append(
        Extension(
            str(str(file.with_suffix('')).replace("/", ".")),
            [file],
            include_dirs=[np.get_include(), 'deps'],
            extra_compile_args=['-O3', '-w', '-std=c++11', '-DEIGEN_NO_MALLOC', '-DNDEBUG', '-fopenmp'],
            extra_link_args=['-fopenmp'],
        )
    )


setuptools.setup(
    name="autoregressive",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)
