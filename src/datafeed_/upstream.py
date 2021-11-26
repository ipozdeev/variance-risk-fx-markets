import os.path

import pandas as pd
import numpy as np
from dotenv import load_dotenv, find_dotenv

from optools.noarbitrage import covered_interest_parity
from downstream import (get_options_data, get_fx_rates, get_interest_rates)

load_dotenv(find_dotenv())


def save_prepared_data() -> None:

    TAU = 1/12

    data_opt = get_options_data()
    data_fx = get_fx_rates()
    data_ir = get_interest_rates()

    # ffill interest rates
    data_ir = data_ir \
        .pivot(index="date", columns="currency", values="value") \
        .resample("B").last() \
        .ffill(limit=10) \
        .reset_index().melt(id_vars="date") \
        .dropna()

    # merge stuff together
    data = pd.merge(
        data_opt,
        data_fx,
        on=["base", "counter", "date"], how="inner"
    )

    # space filler
    data.insert(0, "rf", np.nan)
    data.insert(0, "div_yield", np.nan)

    # fill ir, prioritizing the more trustworty rates
    for c in ["usd", "eur", "gbp", "cad", "jpy",
              "aud", "nzd", "chf", "sek", "nok"]:
        # rf relates to the counter currency
        this_rf = data.loc[:, "rf"].fillna(
            pd.merge(data,
                     data_ir.query(f"currency == '{c}'"),
                     how="left",
                     left_on=["counter", "date"],
                     right_on=["currency", "date"]).loc[:, "value"]
        )

        # div yield relates to the base currency
        this_div_yield = data.loc[:, "div_yield"].fillna(
            pd.merge(data,
                     data_ir.query(f"currency == '{c}'"),
                     how="left",
                     left_on=["base", "date"],
                     right_on=["currency", "date"]).loc[:, "value"]
        )

        this_r = pd.concat((this_rf, this_div_yield), axis=1,
                           keys=["rf", "div_yield"])

        # fill those rows that have both rf and div_yield missing
        idx_missing = data[["div_yield", "rf"]].isnull().all(axis=1)
        data.loc[idx_missing, ["div_yield", "rf"]] = \
            data.loc[idx_missing, ["div_yield", "rf"]].fillna(this_r)

    # fill missing values by no arbitrage between forward and spot
    filled = pd.DataFrame.from_records(
        data.apply(lambda row: covered_interest_parity(
            **row[["spot", "forward", "rf", "div_yield"]],
            tau=TAU
        ), axis=1)
    ).loc[:, ["rf", "div_yield"]]

    data.fillna(filled, inplace=True)

    data.to_feather(
        os.path.join(os.environ.get("LOCAL_DATA_PATH"),
                     "prepared-data.ftr")
    )
    

if __name__ == '__main__':
    save_prepared_data()
