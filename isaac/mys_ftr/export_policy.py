"""Export frozen MYS actor and nominal training robot; verify against JAX."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
os.environ['JAX_PLATFORMS']='cpu'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
import xml.etree.ElementTree as ET
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parent

def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--mys-repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args()
    ROOT=args.output.resolve();ROOT.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.mys_repo.resolve()))
    from loco_mujoco import TaskFactory, PATH_TO_MODELS
    from loco_mujoco.algorithms import PPOJax
    source=args.checkpoint.resolve()
    with source.open('rb') as f:saved=pickle.load(f)
    conf,agent=PPOJax.load_agent(source)
    params=saved['agent_state']['train_state']['params']
    stats=saved['agent_state']['train_state']['run_stats']['RunningMeanStd_0']
    ids=np.asarray(conf.network.actor_obs_ind,dtype=np.int32)
    assert len(ids)==91 and conf.network.action_dim==23, 'Expected Booster T1 FT-R (91 actor observations, 23 actions)'
    arrays={'mean':np.asarray(stats['mean'])[ids], 'var':np.asarray(stats['var'])[ids]}
    for i in range(4):
        for k in ['kernel','bias']:
            arrays[f'{k}_{i}']=np.asarray(params['FullyConnectedNet_0'][f'Dense_{i}'][k])
    from policy import Actor
    actor=Actor(arrays)
    rng=np.random.default_rng(20261022)
    obs=np.asarray(stats['mean'])+rng.normal(size=(128,len(stats['mean']))).astype(np.float32)*np.sqrt(np.asarray(stats['var'])+1e-8)
    pi,_=conf.network.apply({'params':agent.train_state.params,'run_stats':agent.train_state.run_stats},jnp.asarray(obs))
    error=float(np.max(np.abs(np.asarray(pi.mode())-actor(obs[:,ids]))))
    assert error<2e-5,error
    np.savez_compressed(ROOT/'jax_reference.npz',actor_observation=obs[:32,ids],action=np.asarray(pi.mode())[:32])
    np.savez_compressed(ROOT/'actor.npz',**arrays)
    cfg=copy.deepcopy(saved['agent_conf']['config']['experiment']['env_params'])
    (ROOT/'checkpoint_env.json').write_text(json.dumps(cfg,indent=2))
    cfg['domain_randomization_type']='NoDomainRandomization'
    cfg['domain_randomization_params']={}
    cfg['headless']=True
    cfg['goal_params']['visualize_goal']=False
    print('Building nominal checkpoint robot...',flush=True)
    env=TaskFactory.get_factory_cls('RLFactory').make(**cfg)
    model=env.model
    (ROOT/'training_robot.xml').write_text(env.mjspec.to_xml())
    raw=ET.parse(ROOT/'training_robot.xml').getroot()
    compiler=raw.find('compiler')
    compiler.set('meshdir','meshes')
    mesh_source=Path(PATH_TO_MODELS)/'booster_t1'/'assets'
    (ROOT/'meshes').mkdir(exist_ok=True)
    for f in mesh_source.glob('*'):
        if f.is_file():shutil.copy2(f,ROOT/'meshes'/f.name)
    # Only training collisions participate; preserve visual meshes separately.
    active=set()
    pairs=[]
    for i in range(model.npair):
        pair=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(x)) for x in [model.pair_geom1[i],model.pair_geom2[i]]]
        active.update(pair);pairs.append(pair)
    for parent in list(raw.iter()):
        for child in list(parent):
            if child.tag in ['light','camera'] or (child.tag=='geom' and child.get('name')=='floor'):
                parent.remove(child)
    for geom in raw.iter('geom'):
        on=geom.get('name') in active
        geom.set('contype','1' if on else '0');geom.set('conaffinity','1' if on else '0')
    for tag in ['contact','sensor','keyframe']:
        for node in list(raw.findall(tag)):raw.remove(node)
    # mjcf importer 5.1 needs mesh files normalized independently of compiler search paths.
    ET.ElementTree(raw).write(ROOT/'robot_import.xml')
    joints=list(cfg['actuation_spec'])
    jid=[model.joint(j).id for j in joints]
    aid=[next(i for i in range(model.nu) if model.actuator_trnid[i,0]==j) for j in jid]
    feet={}
    for side in ['left','right']:
        site=model.site(side+'_foot')
        feet[side]={'body':model.body(int(site.bodyid[0])).name,'pos':site.pos.tolist(),'quat':site.quat.tolist()}
    meta=dict(actor_sha256=hashlib.sha256((ROOT/'actor.npz').read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              actor_dims=[91,512,256,128,23],numpy_vs_jax_max_abs_error=error,
              actor_indices=ids.tolist(),joint_names=joints,control=cfg['control_params'],
              initial_qpos=cfg['init_state_params']['qpos_init'],physics_dt=float(model.opt.timestep),decimation=cfg['n_substeps'],
              joint_limits=model.jnt_range[jid].tolist(),armature=model.dof_armature[[model.joint(j).dofadr[0] for j in joints]].tolist(),
              joint_damping=model.dof_damping[[model.joint(j).dofadr[0] for j in joints]].tolist(),
              joint_frictionloss=model.dof_frictionloss[[model.joint(j).dofadr[0] for j in joints]].tolist(),
              torque_limits=np.stack([np.maximum(model.actuator_ctrlrange[aid,0],model.jnt_actfrcrange[jid,0]),np.minimum(model.actuator_ctrlrange[aid,1],model.jnt_actfrcrange[jid,1])],axis=1).tolist(),feet=feet,
              native_total_mass=float(model.body_mass.sum()),native_body_masses={model.body(i).name:float(model.body_mass[i]) for i in range(1,model.nbody)},
              native_collision_pairs=pairs,no_training=True,normalization_frozen=True)
    (ROOT/'export.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:meta[k] for k in ['checkpoint_sha256','actor_dims','numpy_vs_jax_max_abs_error','native_total_mass','feet']},indent=2),flush=True)

if __name__=='__main__':main()
