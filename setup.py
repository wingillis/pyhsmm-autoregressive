import numpy as np
import setuptools
from setuptools.extension import Extension
from Cython.Build import cythonize
from pathlib import Path

# Eigen headers are vendored in deps/Eigen (see deps/README.mkd). If they
# are missing, fall back to downloading the pinned 3.3.7 archive.
EIGEN_VERSION = "3.3.7"
EIGEN_SHA256 = "d56fbad95abf993f8af608484729e3d87ef611dd85b3380a8bad1d5cbc373a57"
EIGEN_URL = f"https://gitlab.com/libeigen/eigen/-/archive/{EIGEN_VERSION}/eigen-{EIGEN_VERSION}.tar.gz"


def ensure_eigen():
    eigenpath = Path("deps") / "Eigen"
    if eigenpath.exists():
        return
    import hashlib
    import shutil
    import tarfile
    import urllib.request

    deps = Path("deps")
    deps.mkdir(parents=True, exist_ok=True)
    eigentarpath = deps / "Eigen.tar.gz"
    print(f"Eigen headers not found; downloading {EIGEN_URL} ...")
    req = urllib.request.Request(EIGEN_URL, headers={"User-Agent": "autoregressive-build/1.0"})
    data = urllib.request.urlopen(req, timeout=60).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != EIGEN_SHA256:
        raise RuntimeError(
            f"Eigen archive checksum mismatch: {digest} != {EIGEN_SHA256}"
        )
    eigentarpath.write_bytes(data)
    with tarfile.open(eigentarpath, "r") as tar:
        tar.extractall(deps)
    shutil.move(deps / f"eigen-{EIGEN_VERSION}" / "Eigen", eigenpath)
    print("...done!")


ensure_eigen()

extensions = []

for file in Path("autoregressive").glob("**/*.pyx"):
    extensions.append(
        Extension(
            str(file.with_suffix("")).replace("/", "."),
            sources=[file],
            include_dirs=["deps", np.get_include()],
            extra_compile_args=[
                "-O3",
                "-std=c++11",
                "-DEIGEN_NO_MALLOC",
                "-DNDEBUG",
                "-w",
                "-fopenmp",
            ],
            extra_link_args=["-fopenmp"],
        )
    )

setuptools.setup(
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
