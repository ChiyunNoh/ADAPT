"""Frozen Booster T1 MYS FT-R deployment in Isaac Sim 5.1; native GT only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA','YES')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.stdout.reconfigure(line_buffering=True)
ROOT=Path(__file__).resolve().parent
# Keep the frozen actor and gait clock at the trained 50 Hz.
PHYSICS_DT=.001
POLICY_DT=.02
DECIMATION=round(POLICY_DT/PHYSICS_DT)
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--headless',action='store_true')
p.add_argument('--render',action='store_true',help='Exercise rendering even with a hidden window')
p.add_argument('--seconds',type=float,default=25.)
p.add_argument('--stride',type=float,default=.25)
p.add_argument('--mode',choices=['stand','walk','schedule'],default='walk')
p.add_argument('--seed',type=int,default=0)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--convert-only',action='store_true')
p.add_argument('--rebuild',action='store_true')
p.add_argument('--checkpoint',type=Path,help='Optional original MYS pickle: verify its provenance hash; inference uses bundled actor.npz')
p.add_argument('--disable-self-collision',action='store_true',help='Diagnostic only')
args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
from isaacsim import SimulationApp
app=SimulationApp({'headless':args.headless,'width':1280,'height':720,'anti_aliasing':0})
import numpy as np
import omni.usd
import omni.kit.commands
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdPhysics,UsdShade,PhysxSchema
from policy import Actor,Footholds,rotation,quat

def convert(meta):
    out=ROOT/'usd'/'t1.usd'
    if out.exists() and not args.rebuild:
        return out
    out.parent.mkdir(exist_ok=True)
    enable_extension('isaacsim.asset.importer.mjcf')
    _,cfg=omni.kit.commands.execute('MJCFCreateImportConfig')
    cfg.set_fix_base(False);cfg.set_import_inertia_tensor(True)
    cfg.set_import_sites(False);cfg.set_self_collision(True)
    cfg.set_make_default_prim(True);cfg.set_create_physics_scene(False)
    cfg.set_make_instanceable(False)
    ok,result=omni.kit.commands.execute('MJCFCreateAsset',mjcf_path=str(ROOT/'robot_import.xml'),
                                      import_config=cfg,prim_path='/T1',dest_path=str(out))
    assert ok,(ok,result)
    stage=Usd.Stage.Open(str(out))
    print('ASSET_VARIANTS',{n:stage.GetDefaultPrim().GetVariantSets().GetVariantSet(n).GetVariantNames() for n in stage.GetDefaultPrim().GetVariantSets().GetNames()})
    stage.GetDefaultPrim().GetVariantSets().GetVariantSet('Physics').SetVariantSelection('PhysX')
    # The 5.1 importer instances collision geometry despite its import flag.
    # Deinstance before authoring contact offsets, materials and filtered pairs.
    for prim in list(stage.Traverse()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    collision=[]
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateLinearDampingAttr().Set(0.);rb.CreateAngularDampingAttr().Set(0.)
            rb.CreateMaxDepenetrationVelocityAttr().Set(2.)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive=UsdPhysics.DriveAPI.Apply(prim,'angular')
            drive.CreateTypeAttr().Set('force');drive.CreateStiffnessAttr().Set(0.);drive.CreateDampingAttr().Set(0.)
            drive.CreateMaxForceAttr().Set(1000.)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            ar=PhysxSchema.PhysxArticulationAPI.Apply(prim)
            ar.CreateEnabledSelfCollisionsAttr().Set(True)
            ar.CreateSolverPositionIterationCountAttr().Set(8)
            ar.CreateSolverVelocityIterationCountAttr().Set(4)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision.append(prim)
            col=PhysxSchema.PhysxCollisionAPI.Apply(prim)
            col.CreateContactOffsetAttr().Set(.002);col.CreateRestOffsetAttr().Set(0.)
    # Match the source's explicit self-collision pairs rather than enabling all bodies.
    allowed={frozenset(pair) for pair in meta['native_collision_pairs'] if 'floor' not in pair}
    for prim in collision:
        disallowed=[o.GetPath() for o in collision if o!=prim and frozenset([prim.GetName(),o.GetName()]) not in allowed]
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(disallowed)
    stage.GetRootLayer().Save()
    (ROOT/'usd'/'conversion.json').write_text(json.dumps(dict(import_result=str(result),
        colliders=[str(x.GetPath()) for x in collision],default_prim=str(stage.GetDefaultPrim().GetPath())),indent=2))
    return out

def main():
    meta=json.loads((ROOT/'export.json').read_text())
    actor_hash=hashlib.sha256((ROOT/'actor.npz').read_bytes()).hexdigest()
    assert actor_hash==meta['actor_sha256'], 'Bundled actor changed; re-export before running'
    if args.checkpoint is not None:
        assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==meta['checkpoint_sha256'], 'Original checkpoint does not match bundled actor provenance'
    actor=Actor(dict(np.load(ROOT/'actor.npz')))
    usd=convert(meta)
    if args.convert_only:
        print('CONVERSION_OK',usd);return
    world=World(physics_dt=PHYSICS_DT,rendering_dt=POLICY_DT,stage_units_in_meters=1.)
    stage=omni.usd.get_context().get_stage()
    light=UsdLux.DomeLight.Define(stage,'/World/Light')
    light.CreateIntensityAttr(1000.)
    world.scene.add_default_ground_plane(static_friction=1.,dynamic_friction=1.,restitution=0.)
    root=stage.DefinePrim('/World/T1','Xform');root.GetReferences().AddReference(str(usd))
    arts=[x for x in Usd.PrimRange(root) if x.HasAPI(UsdPhysics.ArticulationRootAPI) and 'worldBody' not in str(x.GetPath())]
    for prim in Usd.PrimRange(root):
        if 'worldBody' in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.SetActive(False)
    assert len(arts)==1,[str(x.GetPath()) for x in arts]
    bodypaths={x.GetName():str(x.GetPath()) for x in Usd.PrimRange(root) if x.HasAPI(UsdPhysics.RigidBodyAPI)}
    print('ARTICULATION',str(arts[0].GetPath()),'BODIES',bodypaths)
    matprim=stage.DefinePrim('/World/T1Material','Material')
    mat=UsdPhysics.MaterialAPI.Apply(matprim)
    mat.CreateStaticFrictionAttr().Set(1.);mat.CreateDynamicFrictionAttr().Set(1.);mat.CreateRestitutionAttr().Set(0.)
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(matprim),UsdShade.Tokens.strongerThanDescendants,'physics')
    feet={side:RigidPrim(bodypaths[spec['body']],name=side+'_foot',track_contact_forces=True,prepare_contact_sensors=True)
          for side,spec in meta['feet'].items()}
    world.reset()
    view=Articulation(str(arts[0].GetPath()),name='t1');view.initialize()
    if args.disable_self_collision:
        view.set_enabled_self_collisions(np.array([False]))
    (args.output/'usd_diagnostics.json').write_text(json.dumps([
        dict(path=str(x.GetPath()),type=x.GetTypeName(),apis=list(x.GetAppliedSchemas()),instance=x.IsInstance(),proxy=x.IsInstanceProxy())
        for x in Usd.PrimRange(root,Usd.TraverseInstanceProxies())],indent=2))
    for f in feet.values():f.initialize()
    names=list(view.dof_names)
    assert set(names)==set(meta['joint_names']),names
    ids=np.array([names.index(x) for x in meta['joint_names']])
    def dof(v):
        out=np.zeros(23,dtype=np.float32);out[ids]=v;return out
    view.set_gains(kps=np.zeros((1,23)),kds=np.zeros((1,23)))
    view.set_armatures(dof(meta['armature'])[None])
    view.set_max_efforts(np.full((1,23),1000.,dtype=np.float32))
    qinit=np.asarray(meta['initial_qpos'])
    qdof=dof(qinit[7:])
    if args.seed:
        qdof+=dof(np.random.default_rng(420000+args.seed).uniform(-.005,.005,23))
    view.set_world_poses(positions=qinit[:3][None],orientations=qinit[3:7][None])
    view.set_velocities(np.zeros((1,6)))
    view.set_joint_positions(qdof[None]);view.set_joint_velocities(np.zeros((1,23)))
    view.set_joint_position_targets(qdof[None]);view.set_joint_velocity_targets(np.zeros((1,23)))
    view.set_joint_efforts(np.zeros((1,23)))
    # One render refreshes the USD display without advancing the reset state.
    world.render()
    def foot_state():
        positions=[];quats=[];forces=[]
        for side,spec in meta['feet'].items():
            pp,qq=feet[side].get_world_poses();rr=rotation(qq[0])
            positions.append(pp[0]+rr.apply(spec['pos']))
            quats.append(quat(rr*rotation(spec['quat'])))
            forces.append(np.asarray(feet[side].get_net_contact_forces(dt=PHYSICS_DT))[0])
        return np.asarray(positions),np.asarray(quats),np.asarray(forces)
    fp,fq,ff=foot_state()
    goals=Footholds(fp,fq)
    markers=[]
    for side,color in [('left',(0.9,.25,.15)),('right',(.15,.4,.9))]:
        cube=UsdGeom.Cube.Define(stage,'/World/Targets/'+side)
        cube.CreateSizeAttr(1.);cube.CreateDisplayColorAttr([color]);cube.CreateDisplayOpacityAttr([.5])
        xf=UsdGeom.Xformable(cube)
        tr=xf.AddTranslateOp();oriop=xf.AddOrientOp();xf.AddScaleOp().Set(Gf.Vec3f(.185,.08,.004))
        markers.append((tr,oriop))
    control=meta['control'];kp=np.asarray(control['p_gain']);kd=np.asarray(control['d_gain'])
    nominal=np.asarray(control['nominal_joint_positions']);limits=np.asarray(meta['joint_limits'])
    torque_limits=np.asarray(meta['torque_limits']);viscous=np.asarray(meta['joint_damping'])
    masses=dict(zip(view.body_names,np.asarray(view.get_body_masses())[0].tolist()))
    mass_error=max(abs(masses[k]-v) for k,v in meta['native_body_masses'].items() if k in masses)
    initial_pose=np.asarray(view.get_world_poses()[0])[0].copy()
    (args.output/'setup.json').write_text(json.dumps(dict(isaacsim='5.1.0',physics_dt=PHYSICS_DT,decimation=DECIMATION,
        policy_dt=POLICY_DT,state_log_dt=PHYSICS_DT,
        source_physics_dt=meta['physics_dt'],source_decimation=meta['decimation'],
        trace_sampling='pre-physics-step; observation/action/targets held between policy updates',
        velocity_frame='world',torque_units='Nm',joint_order=names,policy_to_dof=ids.tolist(),body_masses=masses,
        mass_error_max_kg=mass_error,native_mass_kg=meta['native_total_mass'],
        checkpoint_sha256=meta['checkpoint_sha256'],no_training=True,normalization_frozen=True,
        root= str(arts[0].GetPath()),bodypaths=bodypaths,initial_position=initial_pose.tolist(),
        native_frictionloss_Nm=meta['joint_frictionloss'],frictionloss_model='explicit opposing torque outside zero velocity; PhysX regularization differs',
        self_collision='explicit source geom pairs filtered in USD'),indent=2))
    print('SETUP_READY',dict(mass_error_max_kg=mass_error,initial_foot_pos=fp.tolist(),initial_root=initial_pose.tolist()))
    log={k:[] for k in ['time','root_pos','root_quat','root_lin_vel','q','dq','omega_world',
                        'foot_pos','foot_quat','contact_force','observation','action','torque',
                        'targets','tilt_deg','command','phase','is_policy_step','policy_index']}
    fall=None;bad=0;wall=time.monotonic();simulation_start=float(world.current_time)
    physics_steps=0;policy_updates=0
    fall_samples=round(.2/PHYSICS_DT)
    total_steps=round(args.seconds/POLICY_DT)*DECIMATION
    for step in range(total_steps):
        if not app.is_running():break
        t=step*PHYSICS_DT
        is_policy_step=step%DECIMATION==0
        # Read fresh PhysX state on every 1 ms tick, including between actor calls.
        pos,ori=view.get_world_poses();pos,ori=np.asarray(pos)[0],np.asarray(ori)[0]
        q=np.asarray(view.get_joint_positions())[0,ids];dq=np.asarray(view.get_joint_velocities())[0,ids]
        velocity=np.asarray(view.get_velocities())[0]
        linear_velocity,omega=velocity[:3],velocity[3:]
        fp,fq,ff=foot_state()
        if is_policy_step:
            start=time.monotonic()
            command=np.array([args.stride,0.,0.,float(t>=2. and args.mode!='stand')])
            if args.mode=='schedule':
                if 12.<=t<16.:command[2]=np.deg2rad(10.)
                if t>=16.:command[3]=0.
            obs=goals.observation(pos,ori,q,dq,omega,fp,fq,command,t)
            action=actor(obs).reshape(-1)
            target=np.clip(nominal+np.clip(action,-control['clip_actions'],control['clip_actions'])*control['action_scale'],limits[:,0],limits[:,1])
            policy_updates+=1
        tilt=float(np.rad2deg(np.arccos(np.clip(rotation(ori).as_matrix()[2,2],-1,1))))
        # Preserve the 0.2 s fall-confirmation duration at the faster sample rate.
        bad=bad+1 if pos[2]<.35 or tilt>60. else 0
        if bad>=fall_samples:
            fall=t-(bad-1)*PHYSICS_DT
        friction=np.asarray(meta['joint_frictionloss'])*np.tanh(dq/.01)
        torque=np.clip(kp*(target-q)-(kd+viscous)*dq-friction,torque_limits[:,0],torque_limits[:,1])
        row=dict(time=t,root_pos=pos,root_quat=ori,root_lin_vel=linear_velocity,q=q,dq=dq,omega_world=omega,
                 foot_pos=fp,foot_quat=fq,contact_force=ff,observation=obs,action=action,torque=torque,
                 targets=goals.pos.copy(),tilt_deg=tilt,command=command,phase=goals.phase,
                 is_policy_step=is_policy_step,policy_index=policy_updates-1)
        for key in log:log[key].append(row[key])
        if step%round(1./PHYSICS_DT)==0:
            print('STATE',json.dumps(dict(time=t,position=pos.tolist(),tilt=tilt,walking=bool(command[3]))))
        view.set_joint_efforts(dof(torque)[None])
        world.step(render=False)
        physics_steps+=1
        if (step+1)%DECIMATION==0:
            goals.advance(action)
            if not args.headless or args.render:
                from isaacsim.core.utils.viewports import set_camera_view
                for side,(tr,oriop) in enumerate(markers):
                    marker_position=goals.pos[side].copy();marker_position[2]=.004
                    tr.Set(Gf.Vec3d(*map(float,marker_position)))
                    qq=goals.orn[side];oriop.Set(Gf.Quatf(float(qq[0]),Gf.Vec3f(*map(float,qq[1:]))))
                set_camera_view(eye=pos+[-2.,-2.,1.],target=pos+[0.,0.,.05])
                world.render()
                if not args.headless:time.sleep(max(0.,POLICY_DT-(time.monotonic()-start)))
        if fall is not None:break
    trace={k:np.asarray(v) for k,v in log.items()}
    assert all(np.isfinite(v).all() for v in trace.values())
    np.savez_compressed(args.output/'trace.npz',**trace)
    (args.output/'goals.json').write_text(json.dumps(goals.events,indent=2))
    elapsed=float(world.current_time)-simulation_start
    result=dict(mode=args.mode,seed=args.seed,stride_m=args.stride,simulated_seconds=elapsed,
                fell=fall is not None,fall_time_s=fall,minimum_height_m=float(trace['root_pos'][:,2].min()),
                peak_tilt_deg=float(trace['tilt_deg'].max()),final_position=trace['root_pos'][-1].tolist(),
                displacement_xy_m=float(np.linalg.norm(trace['root_pos'][-1,:2]-trace['root_pos'][0,:2])),
                command=json.dumps(vars(args),default=str),wall_seconds=time.monotonic()-wall,
                actor_unchanged=hashlib.sha256((ROOT/'actor.npz').read_bytes()).hexdigest()==actor_hash,
                source_checkpoint_unchanged=(hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==meta['checkpoint_sha256']) if args.checkpoint is not None else None)
    result.update(physics_dt=PHYSICS_DT,policy_dt=POLICY_DT,state_log_dt=PHYSICS_DT,
                  decimation=DECIMATION,physics_steps=physics_steps,policy_updates=policy_updates,
                  logged_samples=len(trace['time']),physics_time_error_s=elapsed-physics_steps*PHYSICS_DT)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    print('RESULT',json.dumps(result))

try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    (args.output/'FAILED.txt').write_text(traceback.format_exc())
finally:
    app.close()
