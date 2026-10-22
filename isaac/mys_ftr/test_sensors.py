"""Sensor geometry and scan-clock association checks; no Isaac installation needed."""
import unittest

import numpy as np

from sensors import RadarClock, spherical_to_xyz


class SensorTests(unittest.TestCase):
    def test_spherical_units_and_sensor_axes(self):
        points=spherical_to_xyz(np.array([0.,90.,0.]),np.array([0.,0.,90.]),np.array([2.,3.,4.]))
        np.testing.assert_allclose(points,[[2.,0.,0.],[0.,3.,0.],[0.,0.,4.]],atol=1e-12)

    def test_clock_offset_uses_render_pose_not_arrival_time(self):
        clock=RadarClock()
        for t in [0.,.1,.2]:clock.record(t,[t,0.,0.])
        times=clock.resolve(9400000000,9500000000,np.array([.1,0.,0.]),np.array([.2,0.,0.]))
        np.testing.assert_allclose(times,[.1,.2],atol=1e-12)
        # An unrelated sensor-clock jump must not be silently recalibrated.
        self.assertIsNone(clock.resolve(9410000000,9510000000,np.array([.1,0.,0.]),np.array([.2,0.,0.])))

    def test_stale_reset_pose_is_rejected(self):
        clock=RadarClock()
        for t in [0.,.1,.2]:clock.record(t,[t,0.,0.])
        self.assertIsNone(clock.resolve(9400000000,9500000000,np.array([9.,0.,0.]),np.array([.2,0.,0.])))

    def test_ambiguous_stationary_pose_does_not_guess_offset(self):
        clock=RadarClock()
        for t in [0.,.1,.2]:clock.record(t,[0.,0.,0.])
        self.assertIsNone(clock.resolve(9400000000,9500000000,np.zeros(3),np.zeros(3)))
        self.assertIsNone(clock.offset_s)


if __name__=='__main__':unittest.main()
