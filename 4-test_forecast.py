"""Run with: python -m unittest -v test_forecast.py"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import forecast


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.data = forecast.load_data(forecast.ROOT / 'data' / 'synthetic_daily_demand.csv')

    def test_feature_rows_use_only_the_past(self):
        original = forecast.features(self.data)
        changed = self.data.copy()
        changed.loc[100:, 'units'] = 99999
        assert_frame_equal(original.iloc[:101], forecast.features(changed).iloc[:101])
        self.assertEqual(original['past_7_mean'].iloc[100], self.data['units'].iloc[93:100].mean())

    def test_recursive_horizon_and_finite_nonnegative_predictions(self):
        history = self.data.iloc[:-28].copy()
        model = forecast.fit_model(history)
        pred = forecast.recursive_forecast(model, history, 28)
        self.assertEqual(len(pred), 28)
        self.assertEqual(pred['date'].iloc[0], self.data['date'].iloc[-28])
        self.assertEqual(pred['date'].iloc[-1], self.data['date'].iloc[-1])
        self.assertTrue(np.isfinite(pred['predicted_units']).all())
        self.assertTrue((pred['predicted_units'] >= 0).all())

    def test_future_forecast_is_reproducible(self):
        a = forecast.recursive_forecast(forecast.fit_model(self.data), self.data, 14)
        b = forecast.recursive_forecast(forecast.fit_model(self.data), self.data, 14)
        assert_frame_equal(a, b)

    def test_weekly_naive_does_not_use_test_actuals(self):
        history = self.data.iloc[:-28]
        pred = forecast.weekly_naive(history, 28)
        for week in range(4):
            np.testing.assert_array_equal(pred[week * 7:(week + 1) * 7], history['units'].iloc[-7:])

    def test_missing_date_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.csv'
            self.data.drop(index=10).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                forecast.load_data(path)

    def test_negative_units_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.csv'
            bad = self.data.copy()
            bad.loc[0, 'units'] = -1
            bad.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                forecast.load_data(path)

    def test_mae_has_units_not_accuracy_percent(self):
        self.assertEqual(forecast.score([10, 20], [12, 16])['mae_units'], 3.0)


if __name__ == '__main__':
    unittest.main()
