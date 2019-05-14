import logging

import numpy as np
from scipy import linalg


LOGGER = logging.getLogger(__name__)


def extrapolated_lowess(x_data, y_data, alpha=1, y_std=None):
    r"""Performs a locally-weighted regression (LOWESS) and extrapolation for
    missing dependent-variable values.

    Modified version of the traditional LOWESS model to allow extrapolation of
    the dependent variable, i.e. in the situation where the independent
    variable has more data points than the dependent variable.

    Inverse variance weights can be applied if standard deviation of dependent
    variable is given.

    Loosely based on: `"William S. Cleveland: 'Robust locally weighted
    regression and smoothing scatterplots', Journal of the American Statistical
    Association, December 1979, volume 74, number 368, pp. 829-836."`

    In general, we are solving the system of equations

    .. math::

        \vec{y} = \boldsymbol{X}\vec{\beta} \
                         + \vec{\epsilon}

    where

    .. math::

        \boldsymbol{X} = \begin{bmatrix}
                            1      & x_1    \\
                            1      & x_2    \\
                            \vdots & \vdots \\
                            1   & x_{n_\text{all}}
                        \end{bmatrix}

    and :math:`n_\text{all}` is the total number of independent variable values
    in ``x_data`` and with :math:`n_\text{all} \geq n_\text{observed}`, given
    that :math:`n_\text{observed}` is the length of ``y_data``. Also,
    :math:`\vec{\beta}` is a 2-dimension vector containing an intercept and a
    slope. We solve this for each dependent variable value :math:`x_i`, where
    independent variable values are weighted based on their proximity to
    :math:`x_i`; i.e. all dependent variable predictions will have their own
    model fit. When computing the regression for a value :math:`x_i`, the
    (tri cubic) weight for each observed point :math:`(x_j, y_j)` for
    :math:`j\in[0, 1, ..., n_\text{observed}-1]`, is

    .. math::

        w(x_j) = (1 - \lvert d_{ij} \rvert^3)^3

    where :math:`d_{ij}` is the distance between :math:`x_i` (the point the
    regression is being computed for) and :math:`x_j`, scaled to be between 0
    and 1.

    However, if an ordinary simple linear regression is being performed, when
    ``alpha`` :math:`= 1`, then just one system of equations is solved and all
    dependent variables will have the same model fit.

    Note that in a linear regression, we have

    .. math::

        \vec{\beta} = (\boldsymbol{X}^\mathsf{T}\boldsymbol{X})^{-1}
            \boldsymbol{X}^\mathsf{T}\vec{y}


    Meanwhile in _each_ fit of a LOWESS regression (for each :math:`x_i`) we
    have

    .. math::

        \vec{\beta} = (\boldsymbol{X}^\mathsf{T}\hat{\boldsymbol{X}})^{-1}
            \boldsymbol{X}^\mathsf{T}\hat{\vec{y}}

    Where

    .. math::

        \hat{\boldsymbol{X}} = \begin{bmatrix}
                        w_1      & x_1 w_1   \\
                        w_2      & x_2 w_2   \\
                        \vdots   & \vdots    \\
                        w_{n_\text{all}}   & x_{n_\text{all}} w_{n_\text{all}}
                    \end{bmatrix}

    and

    .. math::

        \hat{\vec{y}} = \begin{bmatrix}
                        y_1 w_1   \\
                        y_2 w_2   \\
                        \vdots    \\
                        y_{n_\text{all}} w_{n_\text{all}}
                    \end{bmatrix}

    Args:
        x_data (numpy.ndarray):
            One-dimensional array of independent variable. Must have a length
            greater than or equal to ``y_data``.
        y_data (numpy.ndarray):
            One-dimensional array of dependent variable. Must have a length
            less than or equal to ``x_data``. It assumed that the order of the
            values in this array lineup with their corresponding values in
            ``x_data``. For example, the 5th element of ``y_data`` goes with
            the 5th element of ``x_data``.
        alpha (float | None, optional):
            proportion of window to include, where
            :math:`0 <`  ``alpha``  :math:`\leq 1`. Defaults to 1 -- producing
            a linear regression.
        y_std (np.ndarray | None, optional);
            Standard deviation of dependent variable. Used to construct inverse
            variance weights.

    Returns:
        numpy.ndarray:
            Predictions of the dependent variable. Values will be given in the
            same order that they were. If ``x_data`` has more values than
            ``y_data``, then the indices outside the length of ``y_data`` will
            be treated as extra independent variable data -- corresponding
            dependent variable predictions will be extrapolated and included in
            order in the output.

    Raises:
        RuntimeError:
            If ``y_data``, ``x_data``, and/or ``y_std`` aren't one-dimensional
            arrays.
        RuntimeError:
            If all observed values for, ``x_data``, the independent variable
            are identical. It is impossible to find a relationship with it.
        RuntimeError:
            If there are any NaNs in ``x_data``. The dependent variable cannot
            have gaps.
        RuntimeError:
            If ``y_data`` and ``y_std`` are not the same length.
        RuntimeError:
            If the window includes less than 3 points. To fix this increase
            ``alpha``.
        RuntimeError:
            If some values of ``x_data`` have a weight of zero. To fix this
            increase ``alpha`` or decrease range of ``x_data``.
    """
    if len(y_data.shape) > 1:
        err_msg = "`y_data` is not one-dimensional"
        LOGGER.error(err_msg)
        raise RuntimeError(err_msg)
    elif len(x_data.shape) > 1:
        err_msg = "`x_data` is not one-dimensional"
        LOGGER.error(err_msg)
        raise RuntimeError(err_msg)
    elif y_std is not None and len(y_std.shape) > 1:
        err_msg = "`y_std` is not one-dimensional"
        LOGGER.error(err_msg)
        raise RuntimeError(err_msg)

    number_observed = len(y_data)
    number_predicted = len(x_data)

    x_observed = x_data[:number_observed]
    LOGGER.debug(
        f"{number_observed} observed data points, "
        f"{number_predicted} predicted data points")

    if all(x_observed == x_observed[0]):
        err_msg = (
            "All observed values for, `x_data`, the independent variable are"
            "identical. It is impossible to find a relationship with it.")
        LOGGER.error(err_msg)
        raise RuntimeError(err_msg)
    elif not np.isfinite(x_data).any():
        err_msg = (
            "There are any NaNs in `x_data`. The dependent variable cannot"
            "have gaps")
        LOGGER.error(err_msg)
        raise RuntimeError(err_msg)
    elif y_std is not None and len(y_std) != number_observed:
        err_msg = "`y_std` must have the same number of elements as `y_data`"
        LOGGER.error(err_msg)
        raise RuntimeError(err_msg)

    # Generate tricubic weights
    global_window = int(np.ceil(alpha * number_observed))
    if global_window < 3:
        err_msg = "The window must include at least 3 points. Increase `alpha`"
        LOGGER.error(err_msg)
        raise RuntimeError(err_msg)
    elif global_window >= number_observed:
        LOGGER.debug("Computing linear regression")
        # If all observed data are in the window, then just use equal weights
        # (i.e. use a linear regression).
        weight = np.ones(number_observed)

        weight = _apply_inverse_variance(weight, y_std)

        y_predictions = _predict(y_data, x_data, x_observed, weight)
    else:
        LOGGER.debug("Computing LOWESS regression")
        # Compute the tri cubic weights for every independent variable value,
        # i.e. x value. Each x_i has a vector of weights each corresponding to
        # all the observed data points. The weights are based on the
        # (normalized to the [0, 1] interval) distance of x value of each
        # observed data point from x_i.
        local_window = [
            np.sort(np.abs(x_observed - x_observed[i]))[global_window]
            for i in range(number_observed)]
        base_weight = np.abs(
            (x_data[:, None] - x_observed[None, :]) / local_window).clip(0, 1)
        weight = (1 - base_weight ** 3) ** 3

        weight = _apply_inverse_variance(weight, y_std)

        # Make sure there's still _some_  weight.
        if not all(weight.sum(axis=1) > 0):
            err_msg = (
                "Some values of `x_data` have a weight of zero. Increase"
                "`alpha` or decrease range of `x_data`")
            LOGGER.error(err_msg)
            raise RuntimeError(err_msg)

        # Loop through values of the independent variable to generate
        # corresponding predictions of the dependent variable.
        y_predictions = np.zeros(number_predicted)
        for val_index in range(number_predicted):
            LOGGER.debug(f"fitting localized regression for x[{val_index}]")
            weight_slice = weight[val_index, :]

            try:
                y_predictions[val_index] = _predict(
                    y_data, x_data[val_index], x_observed, weight_slice)
            except np.linalg.LinAlgError as lin_alg_err:
                LOGGER.error(lin_alg_err)
                raise np.linalg.LinAlgError(lin_alg_err)

    return y_predictions


def _predict(y_data, x_data, x_observed, weight):
    r"""Computes the regression using observed dependent and independent
    variable values and makes predictions of the dependent variable
    corresponding to every given value of the independent variable.

    Puts the matrices in the typical, system of equations form:

    .. math::

        \boldsymbol{A}\vec{x} = \vec{b}

    Where the :math:`x` being solved for is :math:`\vec{\beta|`, i.e. the slope
    and intercept of the regression.

    Args:
        y_data (numpy.ndarray):
            All of the observed/available data for the dependent variable.
        x_data (numpy.ndarray):
            One or more independent variable values for which dependent
            variable values are being predicted for. Can have corresponding
            observed dependent variable values or not.
        x_observed (numpy.ndarray):
            Same shape as ``y_data``. The "observed" independent variable
            values, where each value corresponds to a value of the observed
            dependent variable in ``y_data``.
        weight (numpy.ndarray):
            The same length as ``y_data`` and ``x_observed``. There is a
            weight ``weight[i]``, for each index ``i``, corresponding to the
            observed data point ``y_data[i]``, ``x_observed``, indicating how
            much it will influence the regression.

    Returns:
        numpy.ndarray:
            The dependent variable predicted values.
    """
    b = np.array([
        np.sum(weight * y_data),
        np.sum(weight * y_data * x_observed)
        ])
    A = np.array([
        [np.sum(weight),
         np.sum(weight * x_observed)],
        [np.sum(weight * x_observed),
         np.sum(weight * x_observed * x_observed)]
        ])
    beta = linalg.solve(A, b)
    return beta[0] + beta[1] * x_data


def _apply_inverse_variance(weight, y_std):
    """Apply inverse variance weights if standard deviation of dependent
    variable is given."""
    if y_std is not None:
        LOGGER.debug("Applying inverse variance weights")
        return weight / np.square(y_std)
    else:
        LOGGER.debug("Not applying inverse variance weights")
        return weight
