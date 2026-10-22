"""Optional native Isaac IMU and RTX radar recording; no estimator feedback.

Isaac imports are lazy so the RTX launch settings are available before SimulationApp.
The radar uses NVIDIA's generic WPM DMAT model, not a calibrated hardware model.
"""
import json
from pathlib import Path

import numpy as np


RTX_ARGS = [
    '--/renderer/raytracingMotion/enabled=true',
    '--/rtx/materialDb/syncLoads=1',
    '--/rtx/materialDb/nonVisualMaterialCSV/enabled=false',
    '--/rtx/materialDb/rtSensorNameToIdMap=DefaultMaterial:0;MetalAlloyMaterial:7',
    '--/renderer/raytracingMotion/enableHydraEngineMasking=true',
    "--/renderer/raytracingMotion/enabledForHydraEngines='0,1,2,3,4'",
]


def spherical_to_xyz(azimuth_deg, elevation_deg, distance_m):
    az, el = np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg)
    return np.column_stack((distance_m*np.cos(el)*np.cos(az),
                            distance_m*np.cos(el)*np.sin(az), distance_m*np.sin(el)))


class RadarClock:
    """Associate native scan endpoints with recorded render requests, not receipt time.

    GMO has its own clock and can contain stale frames after a robot reset. An
    offset is accepted only when its native end pose matches one unique render
    request. Both scan endpoints must then match the request history. No fitting
    to Doppler, estimator output or task error is used.
    """
    def __init__(self):
        self.history=[]
        self.offset_s=None

    def record(self,time,position):
        self.history.append((float(time),np.asarray(position).copy()))
        self.history=self.history[-32:]

    def resolve(self,start_ns,end_ns,start_position,end_position):
        if not self.history:return None
        times=np.array([x[0] for x in self.history])
        positions=np.array([x[1] for x in self.history])
        if self.offset_s is None:
            matches=np.flatnonzero(np.linalg.norm(positions-end_position,axis=1)<2e-5)
            if len(matches)!=1:return None
            self.offset_s=end_ns*1e-9-times[matches[0]]
        mapped=np.array([start_ns,end_ns])*1e-9-self.offset_s
        for time,position in zip(mapped,[start_position,end_position]):
            index=np.argmin(np.abs(times-time))
            if abs(times[index]-time)>1e-5 or np.linalg.norm(positions[index]-position)>2e-5:
                return None
        return mapped


def add_reflectors(stage):
    """Static steel posts outside the walking lane; visual/radar geometry only."""
    import omni.kit.commands
    from isaacsim.sensors.rtx.impl.nonvisual_materials import apply_nonvisual_material
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    path='/World/SensorLooks/MetalAlloyMaterial'
    omni.kit.commands.execute('CreateMdlMaterialPrim',mtl_url='OmniPBR.mdl',
                              mtl_name='OmniPBR',mtl_path=path)
    material=stage.GetPrimAtPath(path)
    UsdShade.Shader(stage.GetPrimAtPath(path+'/Shader')).CreateInput(
        'diffuse_color_constant',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.55,.6,.65))
    apply_nonvisual_material(material,'steel')
    posts=[]
    for x in range(0,15,2):
        for y in [-3.,3.]:
            posts.append([float(x),y,.9])
    for i,position in enumerate(posts):
        pole=UsdGeom.Cylinder.Define(stage,f'/World/RadarReflectors/post_{i}')
        pole.CreateRadiusAttr(.12);pole.CreateHeightAttr(1.8)
        UsdGeom.Xformable(pole).AddTranslateOp().Set(Gf.Vec3d(*position))
        UsdShade.MaterialBindingAPI.Apply(pole.GetPrim()).Bind(
            UsdShade.Material(material),UsdShade.Tokens.strongerThanDescendants)
    return posts


class SensorSuite:
    def __init__(self,stage,parent,physics_dt,imu=False,radar=False,radar_hz=10.,
                 radar_translation=(.16,0.,.16),radar_pitch_deg=0.,reflectors=True):
        from isaacsim.core.utils.extensions import enable_extension
        from pxr import Gf

        self.imu=None;self.radar=None;self.start_time=0.
        self.imu_rows=[];self.radar_rows=[];self.last_imu_time=None;self.last_radar_time=None
        self.imu_invalid=0;self.imu_duplicates=0
        self.radar_unmatched=0;self.radar_native_frames=0;self.clock=RadarClock()
        self.radar_period=1./radar_hz
        self.meta=dict(imu_enabled=imu,radar_enabled=radar,policy_feedback='GT; sensors recorded only',
                       units=dict(acceleration='m/s^2',angular_velocity='rad/s',position='m',velocity='m/s'),
                       quaternion_order='wxyz',sensor_axes='x forward, y left, z up')
        if imu:
            enable_extension('isaacsim.sensors.physics')
            from isaacsim.sensors.physics import IMUSensor, _sensor
            self.imu=IMUSensor(prim_path=parent+'/imu_sensor',name='recording_imu',dt=physics_dt,
                               translation=np.zeros(3),orientation=np.array([1.,0.,0.,0.]),
                               linear_acceleration_filter_size=1,angular_velocity_filter_size=1,
                               orientation_filter_size=1)
            self.imu_interface=_sensor.acquire_imu_sensor_interface()
            self.meta['imu']=dict(parent=parent,translation=[0.,0.,0.],quaternion_wxyz=[1.,0.,0.,0.],
                                  frequency_hz=1./physics_dt,specific_force=True,read_gravity=True,
                                  noise='native ideal sensor; no added noise/bias',filter_width=1,
                                  orientation='simulator reference only; not an estimator measurement')
        if radar:
            enable_extension('isaacsim.sensors.rtx')
            import omni.kit.commands
            import omni.replicator.core as rep
            from scipy.spatial.transform import Rotation
            q=Rotation.from_euler('y',radar_pitch_deg,degrees=True).as_quat()[[3,0,1,2]]
            ok,self.radar=omni.kit.commands.execute('IsaacSensorCreateRtxRadar',path='radar_sensor',
                parent=parent,translation=Gf.Vec3d(*radar_translation),orientation=Gf.Quatd(*q))
            if not ok:raise RuntimeError('RTX radar creation failed')
            config={'omni:sensor:WpmDmat:auxOutputType':'BASIC','omni:sensor:tickRate':float(radar_hz),
                    'omni:sensor:WpmDmat:elementsCoordsType':'SPHERICAL',
                    'omni:sensor:WpmDmat:outputFrameOfReference':'SENSOR',
                    'omni:sensor:WpmDmat:scan:s001:elevMode':'FULL_EL',
                    'omni:sensor:WpmDmat:scan:s001:maxRangeM':20.,
                    'omni:sensor:WpmDmat:scan:s001:maxAzAngDeg':60.,
                    'omni:sensor:WpmDmat:scan:s001:maxElAngDeg':30.,
                    'omni:sensor:WpmDmat:scan:s001:raysPerDeg':2.}
            for name,value in config.items():
                attr=self.radar.GetAttribute(name)
                if not attr:raise RuntimeError('Radar schema attribute missing: '+name)
                attr.Set(value)
            self.render_product=rep.create.render_product(self.radar.GetPath(),[32,32],
                name='FT_R_Radar',render_vars=['GenericModelOutput','RtxSensorMetadata'])
            self.annotator=rep.AnnotatorRegistry.get_annotator('GenericModelOutput')
            self.annotator.attach([self.render_product.path])
            posts=add_reflectors(stage) if reflectors else []
            self.meta['radar']=dict(parent=parent,translation=list(radar_translation),quaternion_wxyz=q.tolist(),
                frequency_hz=radar_hz,output_frequency_hz=radar_hz,model='NVIDIA generic RTX WPM DMAT',motion_bvh=True,
                coordinates='sensor-frame Cartesian XYZ; raw spherical coordinates also saved',
                doppler='native rv_ms; positive receding, negative approaching; verify against static-scene GT',
                scalar='native uncalibrated return scalar; not claimed to be CFAR SNR or calibrated RCS',
                noise='native model only; no added noise or synthetic floor returns',
                reflectors=posts,reflector_collision=False,settings=config,
                material_mapping='USD nonvisual base steel; nonempty legacy name map supplied for Isaac 5.1 WPM compatibility',
                launch_settings=RTX_ARGS,
                time='scan end in simulation seconds relative to rollout start; time_start is scan start',
                timestamp_ns='raw RTX internal clock at scan start; not simulation time',
                clock_mapping='constant offset from unique native frame-end pose / render-request association; both endpoint poses and times validated',
                frame_pose_gt='native renderer reference only, excluded from estimator sensor measurements',
                rate_control='RTX renders at 50 Hz; only native scans ending on the 10 Hz simulation-time grid are published (5.1 tickRate alone does not throttle output)')

    def initialize(self,start_time):
        self.start_time=start_time
        if self.imu is not None:self.imu.initialize()

    def read_imu(self,receipt_time):
        if self.imu is None:return
        reading=self.imu_interface.get_sensor_reading(self.imu.prim_path,use_latest_data=True,read_gravity=True)
        if not reading.is_valid:
            self.imu_invalid+=1;return
        timestamp=float(reading.time)
        if timestamp==self.last_imu_time:
            self.imu_duplicates+=1;return
        self.last_imu_time=timestamp
        self.imu_rows.append(dict(time=timestamp-self.start_time,receipt_time=receipt_time,
            specific_force=np.array([reading.lin_acc_x,reading.lin_acc_y,reading.lin_acc_z]),
            angular_velocity=np.array([reading.ang_vel_x,reading.ang_vel_y,reading.ang_vel_z]),
            orientation_wxyz_gt=np.asarray(reading.orientation)[[3,0,1,2]].copy()))

    def before_render(self,time,position,quaternion_wxyz):
        if self.radar is None:return
        from scipy.spatial.transform import Rotation
        rotation=Rotation.from_quat(np.asarray(quaternion_wxyz)[[1,2,3,0]])
        self.clock.record(time,np.asarray(position)+rotation.apply(self.meta['radar']['translation']))

    def read_radar(self,receipt_time):
        if self.radar is None:return
        from isaacsim.sensors.rtx import get_gmo_data
        data=self.annotator.get_data()
        raw=data if isinstance(data,np.ndarray) else data.get('data') if isinstance(data,dict) else None
        if raw is None or np.asarray(raw).size<64:return
        gmo=get_gmo_data(raw)
        if int(gmo.magicNumber)!=0x4E474D4F:return
        timestamp=int(gmo.timestampNs)
        if timestamp==self.last_radar_time:return
        self.last_radar_time=timestamp
        endpoints=self.clock.resolve(int(gmo.frameStart.timestampNs),int(gmo.frameEnd.timestampNs),
                                      np.asarray(gmo.frameStart.posM),np.asarray(gmo.frameEnd.posM))
        if endpoints is None:
            self.radar_unmatched+=1;return
        self.radar_native_frames+=1
        if endpoints[1]>receipt_time+1e-5:raise RuntimeError('Radar data from future simulation time')
        # Isaac 5.1's tickRate does not reliably throttle WPM DMAT render output.
        # Keep short render latency and publish real native frames on the 10 Hz grid.
        if abs(endpoints[1]-round(endpoints[1]/self.radar_period)*self.radar_period)>1e-5:return
        if 'SPHERICAL' not in str(gmo.elementsCoordsType) or 'SENSOR' not in str(gmo.frameOfReference):
            raise RuntimeError('Unsupported radar coordinates: expected SPHERICAL / SENSOR')
        if not any(level in str(gmo.auxType) for level in ['BASIC','EXTRA','FULL']):
            raise RuntimeError('Radar GMO has no Doppler auxiliary data')
        n=int(gmo.numElements)
        # Never access aux pointers on an empty GMO buffer (native bindings can crash).
        if n:
            az,el,distance,velocity,scalar=(np.asarray(getattr(gmo,k))[:n].astype(np.float64)
                                          for k in ['x','y','z','rv_ms','scalar'])
            keep=np.isfinite(az+el+distance+velocity+scalar)&(distance>0.)&(distance<20.)
            az,el,distance,velocity,scalar=(v[keep] for v in [az,el,distance,velocity,scalar])
        else:
            az,el,distance,velocity,scalar=[np.zeros(0) for _ in range(5)]
        self.radar_rows.append(dict(time=endpoints[1],time_start=endpoints[0],receipt_time=receipt_time,
            frame_id=int(gmo.frameId),frame_start_pos_world_gt=np.asarray(gmo.frameStart.posM).copy(),
            frame_end_pos_world_gt=np.asarray(gmo.frameEnd.posM).copy(),
            frame_start_quat_wxyz_gt=np.asarray(gmo.frameStart.orientation)[[3,0,1,2]].copy(),
            frame_end_quat_wxyz_gt=np.asarray(gmo.frameEnd.orientation)[[3,0,1,2]].copy(),
            frame_end_timestamp_ns=int(gmo.frameEnd.timestampNs),
            timestamp_ns=timestamp,xyz=spherical_to_xyz(az,el,distance),velocity=velocity,scalar=scalar,
            azimuth_deg=az,elevation_deg=el,range_m=distance))

    def save(self,output):
        output=Path(output)
        if self.imu is not None:
            keys=['time','receipt_time','specific_force','angular_velocity','orientation_wxyz_gt']
            np.savez_compressed(output/'imu.npz',**{k:np.asarray([row[k] for row in self.imu_rows]) for k in keys})
        if self.radar is not None:
            rows=self.radar_rows
            sizes=np.asarray([len(row['velocity']) for row in rows],dtype=np.int64)
            arrays={k:np.asarray([row[k] for row in rows]) for k in ['time','time_start','receipt_time','timestamp_ns',
                'frame_end_timestamp_ns','frame_id','frame_start_pos_world_gt','frame_end_pos_world_gt',
                'frame_start_quat_wxyz_gt','frame_end_quat_wxyz_gt']}
            arrays['offsets']=np.r_[0,np.cumsum(sizes)]
            for key in ['xyz','velocity','scalar','azimuth_deg','elevation_deg','range_m']:
                arrays[key]=np.concatenate([row[key] for row in rows]) if rows else np.zeros((0,3) if key=='xyz' else 0)
            np.savez_compressed(output/'radar.npz',**arrays)
        self.meta.update(imu_samples=len(self.imu_rows),imu_invalid_queries=self.imu_invalid,
                         imu_duplicate_queries=self.imu_duplicates,radar_frames=len(self.radar_rows),
                         radar_nonempty_frames=sum(bool(len(row['velocity'])) for row in self.radar_rows),
                         radar_unmatched_startup_frames=self.radar_unmatched,radar_clock_offset_s=self.clock.offset_s,
                         radar_native_frames=self.radar_native_frames)
        (output/'sensors.json').write_text(json.dumps(self.meta,indent=2))
        return self.meta
