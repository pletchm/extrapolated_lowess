from itertools import product

import numpy as np
import pytest

from extrapolated_lowess import extrapolated_lowess


def root_mean_square_error(predicted, observed):
    return np.sqrt(((predicted - observed) ** 2).mean())


class TestExtrapolatedLowess:
    """Test suite for `fbd_core.etl.extrapolated_lowess` function"""

    @pytest.mark.parametrize("x_vals, y_vals, y_std", list(product(
            # x values:
            (np.ones([4, 100]),
             np.ones([1, 100]),
             np.ones([100])),
            # y values:
            (np.ones([4, 100]),
             np.ones([1, 100]),
             np.ones([100])),
            # y standard deviation values
            (np.ones([4, 100]),
             np.ones([1, 100]),
             np.ones([100]))
        )))
    def test_not_one_dimensional_arrays(self, x_vals, y_vals, y_std):
        """Ensures that RuntimeError is raised in the situations where
        ``y_data``, ``x_data``, and/or ``y_std`` aren't one-dimensional arrays.
        """
        with pytest.raises(RuntimeError):
            extrapolated_lowess(x_vals, y_vals, alpha=1, y_std=y_std)

    def test_identical_x_values(self):
        """Ensures that RuntimeError is raised in the situations where all
        observed independent variable data is identical"""
        alpha = 0.9
        num_x = 150
        num_y = 100
        x_vals = np.ones(num_x)
        y_vals = (np.sin(x_vals[:num_y]) + 0.3 * np.random.randn(num_y) + 0.5)

        with pytest.raises(RuntimeError):
            extrapolated_lowess(x_vals, y_vals, alpha=alpha)

    def test_nans_in_x_data(self):
        """Ensure that RuntimeError is raised in the situations where there
        NaNs in the independent variable data -- it can't have any gaps"""
        alpha = 0.9
        num_x = 150
        num_y = 100
        x_vals = np.linspace(0, 2 * np.pi, num_x)
        x_vals[45] = np.nan
        y_vals = (np.sin(x_vals[:num_y]) + 0.3 * np.random.randn(num_y) + 0.5)

        with pytest.raises(RuntimeError):
            extrapolated_lowess(x_vals, y_vals, alpha=alpha)

    @pytest.mark.parametrize("num_std_vals", [120, 80])
    def test_std_and_observations_inconsistent(self, num_std_vals):
        alpha = 0.9
        num_x = 150
        num_y = 100
        x_vals = np.linspace(0, 2 * np.pi, num_x)
        y_vals = (np.sin(x_vals[:num_y]) + 0.3 * np.random.randn(num_y) + 0.5)
        y_std = np.ones(num_std_vals)
        with pytest.raises(RuntimeError):
            extrapolated_lowess(x_vals, y_vals, alpha=alpha, y_std=y_std)

    @pytest.mark.parametrize("num_x, alpha", list(product(
            # Number of x values:
            (100,     # without extrapolation
             150),    # with extrapolation
            # Alphas:
            (0.02,    # less than 3 points
             -0.9)))  # negative alpha should definitely fail
        )
    def test_alpha_too_small(self, num_x, alpha):
        """Ensure that RuntimeError is raised in the situations where alpha is
        too small, either preventing extrapolation (i.e. causing some weights
        to be zero) or when the window includes less than 3 points."""
        num_y = 100
        x_vals = np.linspace(0, 2 * np.pi, num_x)
        y_vals = (np.sin(x_vals[:num_y]) + 0.3 * np.random.randn(num_y) + 0.5)
        with pytest.raises(RuntimeError):
            extrapolated_lowess(x_vals, y_vals, alpha=alpha)

    @pytest.mark.parametrize("num_x", [
        100,  # without extrapolation
        150   # with extrapolation
        ])
    def test_rmse_decreases_with_alpha(self, num_x):
        """Ensure that as you approach some optimal alpha the RMSE decreases
        -- meaning the fit is improving"""
        num_y = 100
        x_vals = np.linspace(0, 2 * np.pi, num_x)
        y_vals = (np.sin(x_vals[:num_y]) + 0.3 * np.random.randn(num_y) + 0.5)
        alpha = 1
        rmse_vals = []
        while True:
            try:
                y_pred = extrapolated_lowess(x_vals, y_vals, alpha=alpha)
                y_rmse = root_mean_square_error(
                    y_pred[:num_y], y_vals)
                rmse_vals.append(y_rmse)
                alpha -= 0.1
            except RuntimeError:
                # alpha is too small
                break
            except np.linalg.LinAlgError:
                # alpha is too small, resulted in singular matrix
                break

        assert len(rmse_vals) > 1
        assert (np.diff(np.array(rmse_vals)) < 0).all()

    @pytest.mark.parametrize("alpha", [0.9, 0.75, 0.6])
    def test_extrapolated_values(self, alpha):
        """Ensure that extrapolated dependent variable values are present and
        somewhat plausible."""
        num_x = 150
        num_y = 100
        x_vals = np.linspace(0, 2 * np.pi, num_x)
        y_vals = (np.sin(x_vals[:num_y]) + 0.3 * np.random.randn(num_y) + 0.5)

        y_pred = extrapolated_lowess(x_vals, y_vals, alpha=alpha)

        assert len(y_pred) == num_x
        assert np.isfinite(y_pred).all()
        # At least 3/4 of the extrapolated values are unique
        assert (y_pred[num_y:] != y_pred[-1]).sum() > ((num_y - num_x) * 3 / 4)

    def test_inverted_variance(self):
        """Ensure that using standard deviation to construct inverse variance
        weights works"""
        num_x = 150
        num_y = 100
        x_vals = np.linspace(0, 2 * np.pi, num_x)
        y_vals = (
            np.sin(x_vals[:num_y]) + 0.3 * np.random.randn(3, num_y) + 0.5)

        alpha = 1
        rmse_vals = []
        while True:
            try:
                y_pred_no_std = extrapolated_lowess(
                    x_vals, y_vals.mean(axis=0), alpha=alpha)
                y_pred_with_std = extrapolated_lowess(
                    x_vals, y_vals.mean(axis=0), alpha=alpha,
                    y_std=y_vals.std(axis=0))

                # Ensure the use of standard-deviation has _some_ effect.
                assert not (y_pred_no_std == y_pred_with_std).all()

                y_rmse = root_mean_square_error(
                    y_pred_with_std[:num_y], y_vals)
                rmse_vals.append(y_rmse)
                alpha -= 0.1
            except RuntimeError:
                # alpha is too small
                break
            except np.linalg.LinAlgError:
                # alpha is too small, resulted in singular matrix
                break

        # Ensure that with the use of standard-deviation doesn't prevent the
        # the fit from improving as some optimal alpha the RMSE is approached.
        assert len(rmse_vals) > 1
        assert (np.diff(np.array(rmse_vals)) < 0).all()
