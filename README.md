Clone this with something like
```
git clone git://github.com/mattjj/pyhsmm-autoregressive.git autoregressive
```

The compiled code component is optional, but currently requires GCC (for its OpenMP support). Recent versions of Clang (3.7.0 and later) also support OpenMP but require the flag `-fopenmp=libomp`. See [Issue #4](https://github.com/mattjj/pyhsmm-autoregressive/issues/4).

Make sure `libomp` is installed when compiling on linux.

```bash
sudo apt install libomp-dev
```

## Legacy Python 3.7 support

This version requires Python >= 3.12. The final Python 3.7-compatible state
of this repository is preserved on the `py37-legacy` branch (and the
`py37-final` tag):

```bash
pip install "git+https://github.com/wingillis/pyhsmm-autoregressive.git@py37-legacy"
```

The legacy branch is frozen (no new features); the modern branch is the
supported going forward.
