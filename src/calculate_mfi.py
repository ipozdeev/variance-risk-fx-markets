# It is end user's responsibility to provide a working function
# `get_prepared_data` (look up its description and required output in
# `datafeed_.py`). With this function returning the required DataFrame,
# the code below will run.

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv

from utilities import cov_from_vcv
from optools.volsmile import VolatilitySmile
from optools.greeks import strike_from_delta

from datafeed_.downstream import get_prepared_data, get_fx_opt_conventions


# depending on the maturity of contracts in `get_prepared_data`, set this:
TAU = 1 / 12

load_dotenv(find_dotenv())


def calculate_mfiv() -> None:
    """Calculate MFIV and store it as feather."""
    # get data as a DataFrame with rows of (d10..d90, rf, spot, forward etc.)
    data = get_prepared_data()
    data = data\
        .dropna(subset=["d10", "d25", "d50", "d75", "d90"], thresh=4)\
        .dropna(subset=["rf", "div_yield"])
    conv = get_fx_opt_conventions().set_index(["base", "counter"])

    mfiv = list()
    for _, row in tqdm(data.iterrows()):
        c_base, c_counter = row["base"], row["counter"]

        # calculate strikes
        vol_by_d = row.filter(regex="d[0-9]{2}").astype(float).dropna()
        delta_ = np.array(vol_by_d.index.str[1:].astype(int)) / 100.0
        vol_ = vol_by_d.values
        strike_ = strike_from_delta(
            delta=delta_, tau=TAU, vol=vol_, is_call=True,
            is_forward=conv.loc[(c_base, c_counter), "forwarddelta"],
            is_premiumadj=conv.loc[(c_base, c_counter), "premiumadjusted"],
            **row[["spot", "forward", "rf", "div_yield"]]
        )

        # setup a volatility smile instance
        smile_k = VolatilitySmile(strike=strike_, vol=vol_, tau=TAU)

        # calculate mfiv
        mfiv_ = smile_k\
            .interpolate(kind='cubic', extrapolate=True)\
            .get_mfivariance(svix=False, forward=row["forward"], rf=row["rf"])

        mfiv.append(mfiv_)

    # save
    res = pd.concat(
        [data[["base", "counter", "date"]],
         pd.Series(mfiv, index=data.index, name="mfiv")],
        axis=1
    )

    res.reset_index(drop=True).to_feather(
        os.path.join(os.environ.get("LOCAL_DATA_PATH"),
                     "mfiv.ftr")
    )


def calculate_mficov() -> None:
    """Calculate MFICov from MFIV and store it as feather."""
    # MFIV data
    data = pd.read_feather(
        os.path.join(os.environ.get("LOCAL_DATA_PATH"),
                     "mfiv.ftr")
    )

    res = dict()

    for dt, dt_group in tqdm(data.groupby("date")):
        vcv_ = cov_from_vcv(
            dt_group.pivot(index="base", columns="counter", values="mfiv"),
            counter_currency="usd"
        )
        res[dt] = vcv_

    vcv = pd.concat(res, names=["date", "base"]) \
        .drop("usd", axis=1).drop("usd", axis=0, level=1) \
        .sort_index(axis=0).sort_index(axis=1)

    vcv.stack(dropna=False).rename("value")\
        .rename_axis(index=["date", "c1", "c2"]).reset_index()\
        .to_feather(
            os.path.join(os.environ.get("LOCAL_DATA_PATH"),
                         "cov-mfi.ftr")
        )


if __name__ == '__main__':
    calculate_mficov()
