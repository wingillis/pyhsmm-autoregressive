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
        # row t of the strided data contains the affine 1 followed by the
        # nlags previous observations, oldest first
        data = np.arange(1, 6, dtype=float)[:, None]

        strided = AR_striding(data, 2)

        assert strided.shape == (3, 3)
        assert np.array_equal(strided[0], [1., 1., 2.])
        assert np.array_equal(strided[1], [1., 2., 3.])
        assert np.array_equal(strided[2], [1., 3., 4.])

    def test_ar_striding_matches_regression(self):
        rng = np.random.RandomState(0)
        data = rng.randn(50, 4)
        strided = AR_striding(data, 3)
        assert strided.shape == (47, 3 * 4 + 1)
        # first column is the affine offset
        assert np.all(strided[:, 0] == 1.)
        # the lagged blocks line up with the shifted data
        for lag in range(3):
            assert np.allclose(strided[:, 1 + lag * 4:1 + (lag + 1) * 4],
                               data[2 - lag:2 - lag + 47])


class ARBigDataGibbsTester(BigDataGibbsTester):
    # BigDataGibbsTester variant that draws a synthetic AR time series
    # from a ground-truth model and recovers it from the strided data.
    def check_big_data(self, setting_idx, hypparam_dict):
        d1 = self.distribution_class(**hypparam_dict)
        d2 = self.distribution_class(**hypparam_dict)

        data = d1.rvs(prefix=self.prefixes[setting_idx],
                      length=self.big_data_size)
        d2.resample(AR_striding(data, self.nlagss[setting_idx]))

        assert self.params_close(d1, d2)

    @abc.abstractproperty
    def prefixes(self):
        pass

    @abc.abstractproperty
    def nlagss(self):
        pass


@attr('AR_MNIW')
class Test_AR_MNIW(ARBigDataGibbsTester):
    distribution_class = d.AutoRegression

    @property
    def hyperparameter_settings(self):
        return [
            dict(nu_0=5., S_0=np.eye(3), M_0=np.zeros((2, 3)),
                 K_0=np.eye(3), affine=True),
            dict(nu_0=6., S_0=np.eye(9), M_0=np.zeros((2, 9)),
                 K_0=np.eye(9), affine=True, nlags=2),
        ]

    @property
    def prefixes(self):
        return [np.zeros((20, 2)), np.zeros((21, 2))]

    @property
    def nlagss(self):
        return [1, 2]

    def params_close(self, d1, d2):
        return np.allclose(d1.A, d2.A) and np.allclose(d1.sigma, d2.sigma)

    def big_data_Gibbs_tests(self):
        for setting_idx, hypparam_dict in enumerate(self.hyperparameter_settings):
            yield self.check_big_data, setting_idx, hypparam_dict
