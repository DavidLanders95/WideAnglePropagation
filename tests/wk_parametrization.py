"""Weickenmeier-Kohl parametrization for Gold (Au).

Extracted from the au_axel_lubk_verification notebook so tests can reuse it.
Ref: Acta Cryst. (1991). A47, 590-597.
"""
import numpy as np


def weickenmeier_kohl_function(k2, parameters):
    """Evaluate scattering potential V(k) from elastic scattering factor f(s)."""
    A, B = parameters
    s2 = k2 / 4.0
    s2_expanded = s2[..., None]
    term = -np.expm1(-B * s2_expanded)
    sum_term = np.sum(A * term, axis=-1)

    with np.errstate(divide="ignore", invalid="ignore"):
        f_s = sum_term / s2
    limit_val = np.sum(A * B)
    f_s = np.where(s2 == 0, limit_val, f_s)

    mott_bethe_factor = 47.87801
    return f_s * mott_bethe_factor


def make_wk_parametrization():
    """Return an abTEM-compatible Parametrization instance for Au (Weickenmeier-Kohl)."""
    from abtem.parametrizations import Parametrization

    class WeickenmeierKohlParametrization(Parametrization):
        def __init__(self):
            super().__init__(parameters={})
            self._functions = {
                "elastic": weickenmeier_kohl_function,
                "projected_scattering_factor": weickenmeier_kohl_function,
            }

        def scaled_parameters(self, symbol, name):
            if "Au" not in symbol:
                raise NotImplementedError("Only Au is implemented.")
            Z = 79
            V = 0.4
            B = np.array([5.493e-01, 1.728e+00, 6.720e+00,
                          2.637e-02, 7.253e-02, 3.546e+01])
            factor = 0.02395 * Z
            a1_val = factor / (3 * (1 + V))
            A = np.array([a1_val, a1_val, a1_val,
                          V * a1_val, V * a1_val, V * a1_val])
            return [A, B]

        def cutoff(self, symbol):
            return 20.0

    return WeickenmeierKohlParametrization()
