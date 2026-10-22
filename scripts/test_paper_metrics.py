"""Checks for the endpoint metrics in ADAPT Appendix D."""
import unittest

import numpy as np

from scripts.evaluate_paper_metrics import endpoint_metrics


class PaperMetricsTests(unittest.TestCase):
    def test_signed_world_y_and_heading_wrap(self):
        start = np.zeros((4, 3))
        end = np.zeros((4, 3))
        start[:, 1] = [10, 5, -2, 3]
        end[:, 1] = [8, 5, 2, 2]
        lateral, yaw = endpoint_metrics(
            start, end, np.deg2rad([170, 0, 0, 0]),
            np.deg2rad([-170, 359, 360, 540]),
        )
        np.testing.assert_allclose(lateral, [2, 0, 4, 1])
        np.testing.assert_allclose(yaw, np.deg2rad([20, 1, 0, 180]), atol=1e-12)

    def test_wrap_and_magnitude_are_per_trial(self):
        zero = np.zeros((2, 3))
        _, yaw = endpoint_metrics(zero, zero, [0, 0], np.deg2rad([350, -350]))
        self.assertAlmostEqual(float(yaw.mean()), np.deg2rad(10))

    def test_translation_of_world_origin_does_not_change_drift(self):
        start = np.array([[1., 3., 0.]])
        end = np.array([[4., 8., 0.]])
        lateral, yaw = endpoint_metrics(start + 100, end + 100, [1.], [1.])
        np.testing.assert_allclose(lateral, [5.])
        np.testing.assert_allclose(yaw, [0.])


if __name__ == '__main__':
    unittest.main()
