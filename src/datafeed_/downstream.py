import pandas as pd
import os


def get_options_data() -> pd.DataFrame:
    """Get FX option contracts (1m maturity) vol, in frac of 1 p.a.

    Output contains
        'base', 'counter' - str - 3-letter ISO
        'date' - pd.Timestamp
        'd10', 'd25' ... 'd90' - float - vol at that delta, in frac of 1 p.a.
    """
    data = pd.read_feather(
        os.path.join(os.environ.get("RESEARCH_DATA_PATH"),
                     "fx",
                     "fx-iv-by-delta-1m-blb-d.ftr")
    )
    data.loc[:, "vol"] /= 100
    data.loc[:, "delta"] = data["delta"].map("d{}".format)
    data.loc[:, "date"] = pd.to_datetime(data.loc[:, "date"])

    data = data\
        .pivot(index=["base", "counter", "date"],
               columns="delta",
               values="vol")\
        .reset_index()

    return data


def get_interest_rates() -> pd.DataFrame:
    """Get interest rates, in fractions of 1 p.a.

    Output contains
        'date' - pd.Timestamp
        'currency' - str - 3-letter ISO
        'value' - float - interest rate, in frac of 1 p.a.
    """
    res = pd.read_feather(
        os.path.join(os.environ.get("RESEARCH_DATA_PATH"),
                     "fixed-income",
                     "ois_d.ftr")
    )
    res = res.query("maturity == '1m'").drop("maturity", axis=1)
    res.loc[:, "value"] = res.loc[:, "value"] / 100
    res.loc[:, "date"] = pd.to_datetime(res.loc[:, "date"])

    return res


def get_fx_rates() -> pd.DataFrame:
    """Spot and fwd (1m maturity) rates.

    Output contains
        'base', 'counter' - str - 3-letter ISO
        'date' - pd.Timestamp
        'forward', 'spot' - float - units of counter currency for unit of base
    """
    res = pd.read_feather(
        os.path.join(os.environ.get("RESEARCH_DATA_PATH"),
                     "fx",
                     "fx-spot-fwd.ftr")
    )
    res = res.query("date > '2008-06-15'")

    return res


def get_fx_opt_conventions() -> pd.DataFrame:
    """Get conventions such as if delta is premium-adj or not etc.

    Output contains
        'base', 'counter' - str - 3-letter ISO
        'ATM' - str - convention for what is considered at-the-money
        'forwarddelta' - bool - if delta is forward delta
        'premiumadjusted' - bool - if delta is adjusted for option premium
    """
    res = pd.read_feather(
        os.path.join(os.environ.get("RESEARCH_DATA_PATH"),
                     "fx",
                     "fx-opt-conventions.ftr")
    )

    return res


def get_prepared_data() -> pd.DataFrame:
    """Get all the data necessary to calculate MFIV.

    Should return a DataFrame where each row contains all the necessary
    data to calculate MFIV for one tuple of (base, counter, date).

    columns:
        - 'base', 'counter': 3-letter ISO (lowercase)
        - 'date': date-like
        - 'div_yield', 'rf': fractions of 1 p.a., only one per row needed,
        as the other can be filled by no arbitrage
        - 'forward', 'spot': units of the counter currency
        - '10bf', '10rr', '25bf', '25rr', 'atm_vol': Bloomberg quotes / 100

    """
    res = pd.read_feather(
        os.path.join(os.environ.get("LOCAL_DATA_PATH"),
                     "prepared-data.ftr")
    )

    return res


def get_at_data() -> pd.DataFrame:
    """[IRRELEVANT] Get an alternative dataset."""
    res = pd.read_feather(
        os.path.join(os.environ.get("RESEARCH_DATA_PATH"),
                     "fx",
                     "icov-mf-at.ftr")
    )

    return res


if __name__ == '__main__':
    get_fx_rates()
