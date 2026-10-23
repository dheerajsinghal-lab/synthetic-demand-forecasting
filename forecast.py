"""Learning demo: forecast one fictional product's daily demand.

Run: python forecast.py
All bundled data are synthetic. No business use or accuracy guarantee.
"""
from pathlib import Path
import json
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
TEST_DAYS = 28
FORECAST_DAYS = 14
FEATURES = ['day_index', 'weekday_sin', 'weekday_cos', 'lag_7', 'lag_14', 'past_7_mean']


def load_data(path):
    """Reject gaps or invalid sales rather than silently repairing the history."""
    df = pd.read_csv(path)
    if set(df.columns) != {'date', 'units', 'data_type'}:
        raise ValueError('Expected exactly date, units, data_type columns.')
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    df['units'] = pd.to_numeric(df['units'], errors='raise')
    if not (df['data_type'] == 'synthetic').all():
        raise ValueError('This learning demo requires explicitly labelled synthetic data.')
    if len(df) < 90 or df['date'].isna().any() or df['date'].duplicated().any():
        raise ValueError('Need at least 90 rows with unique, nonmissing dates.')
    df = df.sort_values('date').reset_index(drop=True)
    if not df['date'].equals(pd.Series(pd.date_range(df['date'].min(), periods=len(df)))):
        raise ValueError('Dates must be consecutive daily observations.')
    if not np.isfinite(df['units']).all() or (df['units'] < 0).any():
        raise ValueError('Units must be finite, nonnegative numbers.')
    return df


def features(df):
    """Every demand feature uses past values, never the current target."""
    x = pd.DataFrame(index=df.index)
    x['day_index'] = (df['date'] - df['date'].iloc[0]).dt.days
    dow = df['date'].dt.dayofweek
    x['weekday_sin'] = np.sin(2 * np.pi * dow / 7)
    x['weekday_cos'] = np.cos(2 * np.pi * dow / 7)
    x['lag_7'] = df['units'].shift(7)
    x['lag_14'] = df['units'].shift(14)
    x['past_7_mean'] = df['units'].shift(1).rolling(7).mean()
    return x[FEATURES]


def fit_model(history):
    x = features(history)
    valid = x.notna().all(axis=1)
    # Fixed learning-demo setting, not selected using the test scores.
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(x.loc[valid], history.loc[valid, 'units'])
    return model


def recursive_forecast(model, history, days):
    """Fixed-origin forecast. Later lags use predictions, not unseen actuals."""
    working = history[['date', 'units']].copy().reset_index(drop=True)
    rows = []
    for _ in range(days):
        next_date = working['date'].iloc[-1] + pd.Timedelta(days=1)
        candidate = pd.concat([working, pd.DataFrame({'date': [next_date], 'units': [np.nan]})], ignore_index=True)
        next_x = features(candidate).iloc[[-1]]
        prediction = max(0.0, float(model.predict(next_x)[0]))
        working = pd.concat([working, pd.DataFrame({'date': [next_date], 'units': [prediction]})], ignore_index=True)
        rows.append({'date': next_date, 'predicted_units': prediction})
    return pd.DataFrame(rows)


def weekly_naive(history, days):
    """Repeat the last observed seven days, without peeking into the test set."""
    last_week = history['units'].iloc[-7:].to_numpy()
    return np.array([last_week[i % 7] for i in range(days)])


def score(actual, prediction):
    return {'mae_units': float(mean_absolute_error(actual, prediction)),
            'rmse_units': float(np.sqrt(mean_squared_error(actual, prediction)))}


def chart(history, test, future):
    ink, teal, gray = '#251f21', '#73a89a', '#8a8486'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'text.color': ink,
                         'axes.labelcolor': ink, 'xtick.color': ink, 'ytick.color': ink})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), layout='constrained')
    fig.suptitle('Demand forecasting | synthetic learning demo', fontsize=19, x=.07, ha='left')
    axes[0].plot(test['date'], test['actual_units'], color=ink, label='Actual (synthetic)', linewidth=1.7)
    axes[0].plot(test['date'], test['predicted_units'], color=teal, linewidth=2.3, label='Ridge forecast')
    axes[0].plot(test['date'], test['weekly_naive_units'], color=gray, linestyle='--', label='Weekly naive')
    axes[0].set_title('28-day holdout: one forecast made before the test period', loc='left', fontsize=12)
    recent = history.tail(35)
    axes[1].plot(recent['date'], recent['units'], color=ink, linewidth=1.7, label='Observed (synthetic)')
    axes[1].plot(future['date'], future['predicted_units'], color=teal, marker='o', markersize=3, linewidth=2.3, label='14-day future forecast')
    axes[1].axvline(history['date'].iloc[-1], color=gray, linestyle=':', label='History ends')
    axes[1].set_title('Final model: refitted on all 365 observed days', loc='left', fontsize=12)
    for ax in axes:
        ax.set_ylabel('Units per day')
        ax.grid(axis='y', color='#eae9ea')
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=9, loc='upper left')
        ax.tick_params(axis='x', rotation=20, labelsize=9)
        ax.margins(y=.22)
    fig.supxlabel('Demo dates only. No prediction intervals or real-world validation.', fontsize=10)
    fig.savefig(OUT / 'forecast_chart.png', dpi=170, facecolor='white')
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    data = load_data(ROOT / 'data' / 'synthetic_daily_demand.csv')
    train, holdout = data.iloc[:-TEST_DAYS].copy(), data.iloc[-TEST_DAYS:].copy()
    model = fit_model(train)
    test = recursive_forecast(model, train, TEST_DAYS)
    test['actual_units'] = holdout['units'].to_numpy()
    test['weekly_naive_units'] = weekly_naive(train, TEST_DAYS)
    test['data_type'] = 'synthetic'
    metrics = {'data_type': 'synthetic', 'observations': len(data),
               'train_observations': len(train), 'train_feature_rows': len(train) - 14,
               'holdout_days': TEST_DAYS, 'forecast_days': FORECAST_DAYS,
               'train_end': str(train['date'].iloc[-1].date()),
               'holdout_start': str(holdout['date'].iloc[0].date()),
               'holdout_end': str(holdout['date'].iloc[-1].date()),
               'evaluation': 'fixed-origin recursive, no holdout actuals used as features',
               'ridge': score(test['actual_units'], test['predicted_units']),
               'weekly_naive': score(test['actual_units'], test['weekly_naive_units']),
               'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                            'pandas': pd.__version__, 'scikit-learn': sklearn.__version__,
                            'matplotlib': matplotlib.__version__}}
    final_model = fit_model(data)
    future = recursive_forecast(final_model, data, FORECAST_DAYS)
    future['data_type'] = 'synthetic'
    test.to_csv(OUT / 'holdout_predictions.csv', index=False, float_format='%.4f')
    future.to_csv(OUT / 'future_forecast.csv', index=False, float_format='%.4f')
    (OUT / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    chart(data, test, future)
    print('SYNTHETIC LEARNING DEMO - not real business demand')
    print(json.dumps(metrics, indent=2))
    print('\nNext 14 days (expected units, not guaranteed integer orders):')
    print(future[['date', 'predicted_units']].to_string(index=False, float_format=lambda v: f'{v:.1f}'))
    print('\nSaved CSVs, metrics and chart to:', OUT)


if __name__ == '__main__':
    main()
