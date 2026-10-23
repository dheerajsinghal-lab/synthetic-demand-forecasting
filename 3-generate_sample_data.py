"""Generate fictional daily sales for one fictional product. No external data."""
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    out = Path(__file__).resolve().parent / 'data'
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    dates = pd.date_range('2025-10-24', periods=365, freq='D')
    t = np.arange(len(dates))
    # An invented trend, weekly shopping pattern, occasional demand shocks and noise.
    # The forecasting model is NOT given this generating formula.
    weekday_effect = np.array([-10, -7, -4, 0, 8, 25, 18])[dates.dayofweek]
    shocks = np.where(t % 31 == 0, 22, 0)
    units = np.maximum(0, np.rint(105 + .08 * t + weekday_effect + shocks + rng.normal(0, 8, len(dates)))).astype(int)
    df = pd.DataFrame({'date': dates, 'units': units, 'data_type': 'synthetic'})
    df.to_csv(out / 'synthetic_daily_demand.csv', index=False)
    print(f'Created {len(df)} synthetic rows. No real sales/customer/bank data.')


if __name__ == '__main__':
    main()
