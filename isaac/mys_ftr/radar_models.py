"""TI-inspired point-cloud measurement model and empirical surface-return sampler.

These model detections, not TI ADC/chirp processing. Empirical input is loaded
locally and identified by hash; synthetic points are kept separate from RTX data.
"""
import hashlib
from pathlib import Path

import numpy as np


TI_NOISE = dict(range_std_m=.01, velocity_std_m_s=.015, azimuth_std_deg=2.,
                elevation_std_deg=3., range_bin_m=.0488, velocity_bin_m_s=.0762)
TI_PRESET = {
    'omni:sensor:tickRate':10.,
    'omni:sensor:WpmDmat:cfarMode':'2D',
    'omni:sensor:WpmDmat:scan:s001:elevMode':'FULL_EL',
    'omni:sensor:WpmDmat:scan:s001:binsFromSpec':True,
    'omni:sensor:WpmDmat:scan:s001:maxRangeM':5.,
    'omni:sensor:WpmDmat:scan:s001:maxAzAngDeg':60.,
    'omni:sensor:WpmDmat:scan:s001:maxElAngDeg':60.,
    'omni:sensor:WpmDmat:scan:s001:raysPerDeg':4.,
    'omni:sensor:WpmDmat:scan:s001:rangeResM':.065,
    'omni:sensor:WpmDmat:scan:s001:velResMps':.08,
    'omni:sensor:WpmDmat:scan:s001:maxVelMpsSequence':[2.48,2.48],
    'omni:sensor:WpmDmat:scan:s001:boreAzResDeg':30.,
    'omni:sensor:WpmDmat:scan:s001:boreElResDeg':38.,
    'omni:sensor:WpmDmat:scan:s001:cfarRnT':8,
    'omni:sensor:WpmDmat:scan:s001:cfarRnG':4,
    'omni:sensor:WpmDmat:scan:s001:cfarVnT':8,
    'omni:sensor:WpmDmat:scan:s001:cfarVnG':4,
    'omni:sensor:WpmDmat:scan:s001:cfarOffset':1.,
    # Add the empirical measurement noise exactly once, after RTX/surface merging.
    'omni:sensor:WpmDmat:scan:s001:azimuthRadNoiseMean':0.,
    'omni:sensor:WpmDmat:scan:s001:azimuthRadNoiseSDev':0.,
    'omni:sensor:WpmDmat:scan:s001:rangeNoiseMean':0.,
    'omni:sensor:WpmDmat:scan:s001:rangeNoiseSDev':0.,
}


def xyz_to_spherical(xyz):
    xyz=np.asarray(xyz,dtype=float).reshape(-1,3)
    distance=np.linalg.norm(xyz,axis=1)
    az=np.rad2deg(np.arctan2(xyz[:,1],xyz[:,0]))
    el=np.rad2deg(np.arcsin(np.clip(xyz[:,2]/np.maximum(distance,1e-12),-1.,1.)))
    return az,el,distance


def spherical_to_xyz(az,el,distance):
    a,e=np.deg2rad(az),np.deg2rad(el)
    return np.column_stack((distance*np.cos(e)*np.cos(a),distance*np.cos(e)*np.sin(a),distance*np.sin(e)))


def measurement_model(xyz,velocity,rng):
    """Independent Gaussian perturbations, followed by range/Doppler quantization."""
    az,el,distance=xyz_to_spherical(xyz)
    distance=np.maximum(distance+rng.normal(0.,TI_NOISE['range_std_m'],distance.shape),0.)
    velocity=np.asarray(velocity)+rng.normal(0.,TI_NOISE['velocity_std_m_s'],len(velocity))
    az=az+rng.normal(0.,TI_NOISE['azimuth_std_deg'],az.shape)
    el=el+rng.normal(0.,TI_NOISE['elevation_std_deg'],el.shape)
    distance=np.round(distance/TI_NOISE['range_bin_m'])*TI_NOISE['range_bin_m']
    velocity=np.round(velocity/TI_NOISE['velocity_bin_m_s'])*TI_NOISE['velocity_bin_m_s']
    return spherical_to_xyz(az,el,distance),velocity,az,el,distance


class FloorReturnModel:
    """Sample empirical directions, then intersect the actual static PhysX scene.

    Robot/dynamic first hits reject the ray, so occluded ground is not filled in.
    Counts are a requested sample budget; FOV, range and occlusion can reduce it.
    """
    def __init__(self,path,seed=0):
        self.path=Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f'Empirical floor model missing: {path}; pass --radar-floor-model or --no-synthetic-floor')
        with np.load(self.path,allow_pickle=False) as f:
            self.distance=np.array(f['d'],dtype=float)
            self.az=np.deg2rad(np.array(f['az'],dtype=float))
            self.counts=np.array(f['counts'],dtype=np.int64)
            self.intensity=np.array(f['intensity'],dtype=float)
            self.height=float(f['height'])
        if (self.distance.ndim!=1 or self.az.shape!=self.distance.shape or not len(self.distance)
            or not len(self.counts) or np.any(self.counts<0) or not len(self.intensity)
            or self.height<=0 or np.any(self.distance<=0)
            or not all(np.isfinite(x).all() for x in [self.distance,self.az,self.intensity])):
            raise ValueError('Invalid empirical floor-return distribution')
        self.depression=np.arctan2(self.height,self.distance)
        self.rng=np.random.default_rng(seed)
        self.meta=dict(filename=self.path.name,sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
            source='existing 0916 run_02 empirical floor distribution',samples=len(self.distance),
            frames=len(self.counts),reference_height_m=self.height,requested_points_mean=float(self.counts.mean()),
            direction_jitter='none beyond empirical sampling; TI measurement noise applied once after merging',
            static_surface_assumption=True,robot_and_dynamic_first_hits='reject ray; do not see through occluders')

    def sample(self,position,rotation,cast,robot_prefix='/World/T1',ray_origin=None):
        n=int(self.rng.choice(self.counts))
        origin=np.asarray(position) if ray_origin is None else np.asarray(ray_origin)
        yaw=np.arctan2(rotation[1,0],rotation[0,0])
        cy,sy=np.cos(yaw),np.sin(yaw)
        xyz=[];intensity=[];robot_hits=0;dynamic_hits=0;attempts=0
        for index in self.rng.integers(len(self.distance),size=3*n+3 if n else 0):
            if len(xyz)>=n:break
            attempts+=1
            a,dep=self.az[index],self.depression[index]
            level=np.array([np.cos(dep)*np.cos(a),np.cos(dep)*np.sin(a),-np.sin(dep)])
            direction=np.array([cy*level[0]-sy*level[1],sy*level[0]+cy*level[1],level[2]])
            hit=cast(origin,direction,5.)
            if hit is None:continue
            point,body,dynamic=hit
            if body.startswith(robot_prefix):robot_hits+=1;continue
            if dynamic:dynamic_hits+=1;continue
            p=rotation.T@(np.asarray(point)-position)
            az,el,distance=xyz_to_spherical(p[None])
            if not (0.<distance[0]<=5. and abs(az[0])<=60. and abs(el[0])<=60.):continue
            xyz.append(p);intensity.append(float(self.rng.choice(self.intensity)))
        return dict(xyz=np.asarray(xyz,dtype=float).reshape(-1,3),scalar=np.asarray(intensity),
                    requested=n,visible=len(xyz),robot_occluded=robot_hits,dynamic_rejected=dynamic_hits,attempts=attempts)
