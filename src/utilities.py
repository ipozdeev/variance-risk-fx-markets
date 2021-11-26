import pandas as pd
import numpy as np

from foolbox.backtesting.weights import rescale_weights


def cov_from_vcv(vcv, counter_currency) -> pd.DataFrame:
    """Calculate covariance between rates against a common counter_currency.

    Buids on:
    var[ab] = var[ac - bc] = var[ac] + var[bc] - 2cov[ac, bc]
    cov[ac, bc] = -1/2 (var[ab] - var[ac] - var[bc])

    Parameters
    ----------
    vcv : pd.DataFrame
        where entry (p, q) references variance of PQ pair
    counter_currency : str
        must be in index and columns of `vcv`

    """
    # reindex to have the same currencies in idx and cols
    idx = vcv.columns.union(vcv.index)
    vcv_ = vcv.reindex(columns=idx, index=idx)

    # extract var[ac], var[bc] (assumed to be the same as var[ca], var[cb])
    var_cc = vcv_[counter_currency].fillna(vcv_.loc[counter_currency])

    # calculate from the var/cov relation
    res = -0.5 * (vcv_.sub(var_cc, axis=0).sub(var_cc, axis=1))

    # fill diagonal w/variances, reflect to have symmetry
    np.fill_diagonal(res.values, var_cc.values)
    res.fillna(res.T, inplace=True)

    return res


def betas_from_covmat(covmat, weight, exclude_self=False, dropna=False):
    """Calculate the beta of variables w.r.t. their linear combination.

    Given the covariance matrix of variables, calculates the beta of variable
    p with respect to an index comprised of these variables:

        v_i = a + b*index + eps,
        index = V * weights,
        V = [v_1, ..., v_i, ..., v_k]

    The weight of variable i itself can be zero.

    Parameters
    ----------
    covmat : pandas.DataFrame
        covariance matrix
    weight : pandas.Series
        weights of each asset in the linear combination
    exclude_self : boolean
        True if the asset for which the beta is calculated should be
        excluded from calculation of the index
    dropna : bool
        True to remove rows/columns with NA values and rescale `weights`

    Returns
    -------
    betas : pandas.Series
        of calculated betas (ordering corresponds to columns of `covmat`)

    """
    # leverage
    lvg = "zero" if (np.sign(weight).diff() == 0).all() else "net"

    if dropna:
        nan_count = covmat.isnull().sum().sort_values(ascending=False)
        genr = iter(nan_count.index)
        while covmat.isnull().any().any():
            c_ = next(genr)
            covmat = covmat.drop(c_, axis=0).drop(c_, axis=1)

        weight = weight.reindex(covmat.index)
        weight = weight/weight.sum()

    if exclude_self:
        betas = pd.Series(index=weight.index)

        for c in weight.index:
            tmp_wght = weight.copy()

            # set weight to 0
            tmp_wght.loc[c] = 0.0

            # recalculate weights
            tmp_wght = rescale_weights(tmp_wght, leverage=lvg)

            # beta
            this_num = covmat.dot(tmp_wght)
            this_denom = tmp_wght.dot(covmat.dot(tmp_wght))

            betas.loc[c] = this_num.loc[c]/this_denom

    else:
        this_num = covmat.dot(weight)
        this_denom = weight.dot(covmat.dot(weight))
        betas = this_num/this_denom

    # reindex back
    res = betas.reindex(covmat.columns)

    return res
