"""Deterministic algebra checks, not localization experiments; standard library only.

See DEPLOYMENT_DRIFT_DERIVATION.md, committed before this implementation.
No data/model inputs, learned parameters, RNG, optimizer, or performance metrics.
"""
import json
import math


def phi(z):
    return math.exp(-z * z / 2) / math.sqrt(2 * math.pi)


def cdf(z):
    return math.erfc(-z / math.sqrt(2)) / 2


def encoded_mean(mu, variance=0.0, h=1.0, theta=-85.0, s=5.0, c=-110.0):
    q = math.sqrt(s * s + variance)
    z = (mu - theta) / q
    return c + h * ((mu - c) * cdf(z) + variance / q * phi(z))


def encoded_derivative(mu, variance):
    q = math.sqrt(25 + variance)
    z = (mu + 85) / q
    return cdf(z) + (mu + 110) / q * phi(z) - variance * z / q**2 * phi(z)


def main():
    checks = {}
    # Total variance of a random component, versus measuring their average.
    mix_var = 2**2 + ((-80 + 60)**2 + (-40 + 60)**2) / 2
    assert mix_var == 404 and mix_var != 4
    checks["random_location_not_average_measurement"] = {
        "mixture_variance": mix_var, "linear_average_variance": 4,
        "density_ratio_at_minus60_linear_over_mixture": math.exp(50),
        "extra_field_uncertainty_if_C_is_identity": {"random_location": 1, "average": 0.5},
    }

    true_bias, observation, wrongly_associated_old_value = 5, -75, -60
    estimated_bias = observation - wrongly_associated_old_value
    assert estimated_bias == -15 and estimated_bias != true_bias
    checks["wrong_association_biases_constant"] = {"true": true_bias, "estimated": estimated_bias}
    assert -80 + 20 == -60 + 0
    checks["occupancy_and_drift_nonidentifiable"] = {
        "same_observation": -60, "possible_biases": [20, 0],
    }

    observations = [-75, -55]
    old = [-80, -60]
    mean_residual = lambda candidates: sum(x - f for x, f in zip(observations, candidates)) / 2
    ordinary = mean_residual(old)
    permuted = mean_residual(old[::-1])
    changed_occupancy = mean_residual([-60, -60])
    assert ordinary == permuted == 5 and changed_occupancy == -5
    checks["row_permutation_only_invariance"] = {
        "original": ordinary, "row_permuted": permuted, "changed_column_mass": changed_occupancy,
    }

    shifts = {str(mu): encoded_mean(mu + 5) - encoded_mean(mu) for mu in [-95, -85, -65]}
    # Compare to the physical shift; an earlier hand-estimated <2 bound was wrong.
    assert 0 < shifts["-95"] < 5 and shifts["-85"] > 5 and abs(shifts["-65"] - 5) < 0.01
    errors = []
    for mu in [-100.0, -85.0, -65.0]:
        for variance in [0.0, 25.0, 100.0]:
            eps = 1e-4
            fd = (encoded_mean(mu + eps, variance) - encoded_mean(mu - eps, variance)) / (2 * eps)
            errors.append(abs(fd - encoded_derivative(mu, variance)))
    assert max(errors) < 1e-7
    # Independent quadrature of E[c + (L-c)*Phi((L-theta)/s)].
    n, lo, hi = 4000, -9.0, 9.0
    step = (hi - lo) / n
    def integrand(t):
        latent = -85 + 5 * t
        return (-110 + (latent + 110) * cdf((latent + 85) / 5)) * phi(t)
    integral = step / 3 * (integrand(lo) + integrand(hi) + sum(
        (4 if i % 2 else 2) * integrand(lo + i * step) for i in range(1, n)))
    integral_error = abs(integral - encoded_mean(-85, 25))
    assert integral_error < 1e-8
    checks["physical_shift_not_encoded_constant"] = {
        "latent_shift": 5, "encoded_shifts": shifts,
        "threshold_derivative_v0": encoded_derivative(-85, 0),
        "max_derivative_fd_error": max(errors), "quadrature_error": integral_error,
    }

    old_conditional_variance = 1 - 1 / (1 + 0.3)
    new_variance = old_conditional_variance + 4
    assert new_variance > 4
    checks["old_anchor_does_not_pin_future_drift"] = {
        "old_conditional_variance": old_conditional_variance, "new_variance": new_variance,
    }
    iid, correlated = 1 / 20, 0.8 + 0.2 / 20
    assert correlated > 16 * iid
    checks["correlated_repeats"] = {"N": 20, "rho": 0.8, "iid_mean_variance": iid,
                                      "correlated_mean_variance": correlated}
    x, y, context = [1, 3], [4, -2], [20, -7]
    dist = sum((a - b)**2 for a, b in zip(x, y))
    conditioned_dist = sum(((a + c) - (b + c))**2 for a, b, c in zip(x, y, context))
    assert dist == conditioned_dist
    checks["shared_final_feature_offset_cancels"] = {"before": dist, "after": conditioned_dist}
    return {"purpose": "analytic counterexamples only; no UJI model or performance evaluation",
            "checks_passed": len(checks), "checks": checks}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False, allow_nan=False))
