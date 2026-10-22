"""Measurement statistics and surface geometry, including occlusion and slopes."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from radar_models import FloorReturnModel, measurement_model, xyz_to_spherical


class RadarModelTests(unittest.TestCase):
    def model(self,directory):
        path=Path(directory)/'empirical.npz'
        np.savez(path,d=np.array([1.,2.,3.]),az=np.array([-25.,0.,25.]),
                 intensity=np.array([10.,15.,20.]),counts=np.array([12]),height=1.23)
        return FloorReturnModel(path,seed=7)

    def test_noise_scales_and_hardware_bins(self):
        n=60000
        xyz=np.tile([2.,0.,0.],(n,1))
        points,velocity,az,el,distance=measurement_model(xyz,np.zeros(n),np.random.default_rng(42))
        np.testing.assert_allclose(distance/.0488,np.round(distance/.0488),atol=1e-12)
        np.testing.assert_allclose(velocity/.0762,np.round(velocity/.0762),atol=1e-12)
        self.assertAlmostEqual(float(az.std()),2.,delta=.04)
        self.assertAlmostEqual(float(el.std()),3.,delta=.06)
        # A stationary scene should mostly quantize to zero. Do not force the
        # trajectory-specific 24% zero fraction from the original moving bag.
        self.assertGreater(float(np.mean(velocity==0.)),.98)
        np.testing.assert_allclose(np.linalg.norm(points,axis=1),distance,atol=1e-12)

    def test_noise_seed_reproduces_measurements(self):
        xyz=np.array([[2.,.2,-.4],[1.,-.3,.1]])
        first=measurement_model(xyz,np.array([.2,-.3]),np.random.default_rng(9))
        second=measurement_model(xyz,np.array([.2,-.3]),np.random.default_rng(9))
        for a,b in zip(first,second):np.testing.assert_array_equal(a,b)

    def test_downward_mount_hits_actual_flat_surface(self):
        origin=np.array([0.,0.,.85]);rotation=Rotation.from_euler('y',20.,degrees=True).as_matrix()
        self.assertLess(rotation[2,0],0.)
        def cast(p,d,limit):
            t=-p[2]/d[2]
            return (p+t*d,'/World/groundPlane',False) if 0.<t<limit else None
        with tempfile.TemporaryDirectory() as directory:
            result=self.model(directory).sample(origin,rotation,cast)
        self.assertEqual(result['visible'],12)
        world=origin+result['xyz']@rotation.T
        np.testing.assert_allclose(world[:,2],0.,atol=1e-12)
        az,el,distance=xyz_to_spherical(result['xyz'])
        self.assertTrue(np.all(np.abs(az)<=60.) and np.all(np.abs(el)<=60.) and np.all(distance<=5.))

    def test_slope_hits_follow_geometry_instead_of_forcing_z_zero(self):
        origin=np.array([0.,0.,.85]);rotation=Rotation.from_euler('y',20.,degrees=True).as_matrix()
        normal=np.array([-.2,0.,1.])
        def cast(p,d,limit):
            t=(.1-normal@p)/(normal@d)
            return (p+t*d,'/World/slope',False) if 0.<t<limit else None
        with tempfile.TemporaryDirectory() as directory:
            result=self.model(directory).sample(origin,rotation,cast)
        self.assertGreater(result['visible'],0)
        world=origin+result['xyz']@rotation.T
        np.testing.assert_allclose(world[:,2],.2*world[:,0]+.1,atol=1e-12)

    def test_robot_and_dynamic_occluders_do_not_reveal_ground_behind_them(self):
        origin=np.array([0.,0.,.85]);rotation=Rotation.from_euler('y',20.,degrees=True).as_matrix()
        for body,counter in [('/World/T1/leg','robot_occluded'),('/World/moving_box','dynamic_rejected')]:
            with self.subTest(body=body),tempfile.TemporaryDirectory() as directory:
                result=self.model(directory).sample(origin,rotation,lambda p,d,m:(p+.1*d,body,True))
            self.assertEqual(result['visible'],0)
            self.assertEqual(result[counter],result['attempts'])


if __name__=='__main__':unittest.main()
