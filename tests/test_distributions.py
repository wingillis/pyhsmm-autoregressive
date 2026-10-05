import numpy as np
import abc
from matplotlib import pyplot as plt

from nose.plugins.attrib import attr
from pybasicbayes.testing.mixins import BigDataGibbsTester

from autoregressive import distributions as d
from autoregressive.util import AR_striding


@attr('AR_striding')
class TestARStriding:
    def test_ar_striding(self):
        # as_strided with row-major interleaving: strided row i contains
        # data[i], ..., data[i+nlags] flattened (oldest first, target last)
        data = np.arange(1, 6, dtype=float)[:, None]

        strided = AR_striding(data, 2)

        assert strided.shape == (3, 3)
        assert np.array_equal(strided[0], [1., 2., 3.])
        assert np.array_equal(strided[1], [2., 3., 4.])
        assert np.array_equal(strided[2], [3., 4., 5.])

    def test_ar_striding_blocks(self):
        rng = np.random.RandomState(0)
        data = rng.randn(50, 4)
        nlags = 3
        strided = AR_striding(data, nlags)
        assert strided.shape == (50 - nlags, 4 * (nlags + 1))
        # lag blocks are chronological; the final block is the target
        for lag in range(nlags + 1):
            assert np.allclose(
                strided[:, lag * 4:(lag + 1) * 4],
                data[lag:lag + (50 - nlags)])


class ARBigDataGibbsTester(BigDataGibbsTester):
    # Draws a synthetic AR time series from a ground-truth AutoRegression
    # and checks that a fresh model recovers A and sigma from the strided
    # data (the conjugate posterior concentrates on the truth).
    def check_big_data(self, setting_idx, hypparam_dict):
        d1 = self.distribution_class(**hypparam_dict)
        d2 = self.distribution_class(**hypparam_dict)

        nlags = d1.nlags
        rows = [np.zeros(d1.D_out) for _ in range(nlags)]
        for _ in range(self.big_data_size):
            rows.append(d1.rvs(np.asarray(rows[-nlags:]))[0])
        data = np.asarray(rows)
        d2.resample(AR_striding(data, nlags))

        assert self.params_close(d1, d2)

    @abc.abstractproperty
    def distribution_class(self):
        pass

    def params_close(self, d1, d2):
        return np.allclose(d1.A, d2.A, atol=0.05) and \
            np.allclose(d1.sigma, d2.sigma, atol=0.05)

    @property
    def big_data_size(self):
        return 20000


@attr('AR_MNIW')
class Test_AR_MNIW(ARBigDataGibbsTester):
    distribution_class = d.AutoRegression

    @property
    def hyperparameter_settings(self):
        return [
            # nlags=1, affine (D_in = 2*1 + 1)
            dict(nu_0=5., S_0=np.eye(2), M_0=np.zeros((2, 3)),
                 K_0=np.eye(3), affine=True),
            # nlags=2, affine (D_in = 2*2 + 1)
            dict(nu_0=6., S_0=np.eye(2), M_0=np.zeros((2, 5)),
                 K_0=np.eye(5), affine=True),
        ]

    def big_data_Gibbs_tests(self):
        for setting_idx, hypparam_dict in enumerate(self.hyperparameter_settings):
            yield self.check_big_data, setting_idx, hypparam_dict
