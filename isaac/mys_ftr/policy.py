"""NumPy-only deterministic frozen actor and MYS FT-R observation adapter."""
import numpy as np
from scipy.spatial.transform import Rotation

class Actor:
    def __init__(self,arrays):
        self.a={k:np.asarray(v,dtype=np.float32) for k,v in arrays.items()}
    def __call__(self,obs):
        a=self.a
        x=(np.asarray(obs,dtype=np.float32)-a['mean'])/np.sqrt(a['var']+np.float32(1e-8))
        for i in range(4):
            x=x@a[f'kernel_{i}']+a[f'bias_{i}']
            if i<3:
                x=np.maximum(x,0)+np.expm1(np.minimum(x,0))
        return x.astype(np.float32)

def rotation(q):
    return Rotation.from_quat(np.asarray(q)[[1,2,3,0]])

def quat(r):
    q=r.as_quat()[[3,0,1,2]]
    return q if q[0]>=0 else -q

class Footholds:
    """Port of KeyboardFootPlacement: targets in stance yaw; FT-R in root frame."""
    def __init__(self,foot_pos,foot_quat,phase=0.):
        self.pos=np.asarray(foot_pos).copy();self.orn=np.asarray(foot_quat).copy()
        self.phase=phase;self.swing=0 if phase<.5 else 1;self.cycles=1;self.still=True
        self.last_action=np.zeros(23,dtype=np.float32)
        self.events=[]
    def observation(self,root_pos,root_quat,q,dq,omega_world,foot_pos,foot_quat,command,t):
        root=rotation(root_quat);rots=[rotation(x) for x in foot_quat]
        swing=0 if self.phase<.5 else 1
        stopped=command[3]<.5
        # Match command application at a gait transition, as the source viewer.
        if swing!=self.swing or (t==0.):
            transition=stopped!=self.still
            self.cycles=1 if transition else min(self.cycles+1,3)
            self.still=stopped;self.swing=swing
            sign=1 if swing==0 else -1
            width=.2 if stopped else .1
            cmd=np.zeros(4) if stopped else command
            lateral=sign*max(width,width+sign*cmd[1])
            stance=1-swing
            yaw=rots[stance].as_euler('xyz')[2]
            cy,sy=np.cos(yaw),np.sin(yaw)
            self.pos[swing]=foot_pos[stance]+[cy*cmd[0]-sy*lateral,sy*cmd[0]+cy*lateral,0.]
            self.orn[swing]=quat(Rotation.from_euler('z',yaw+cmd[2]))
            self.events.append(dict(time=t,side=swing,target=self.pos[swing].tolist(),walking=not stopped))
        if self.still and self.cycles>=2:
            targets=[foot_pos[1]+rots[1].apply([0.,.2,0.]),foot_pos[0]+rots[0].apply([0.,-.2,0.])]
            goal=np.r_[root.inv().apply(targets[0]-root_pos),[1,0,0,0],root.inv().apply(targets[1]-root_pos),[1,0,0,0],[0,0]]
        else:
            goal=np.r_[root.inv().apply(self.pos[0]-root_pos),quat(root.inv()*rotation(self.orn[0])),
                       root.inv().apply(self.pos[1]-root_pos),quat(root.inv()*rotation(self.orn[1])),
                       [np.cos(2*np.pi*self.phase),np.sin(2*np.pi*self.phase)]]
        gravity=root.inv().apply([0,0,-1.])
        omega=root.inv().apply(omega_world)
        return np.r_[gravity,q,omega,dq,self.last_action,goal].astype(np.float32)
    def advance(self,action):
        self.last_action=np.asarray(action,dtype=np.float32)
        self.phase=(self.phase+.02)%1.
