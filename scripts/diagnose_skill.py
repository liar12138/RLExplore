"""Privileged low-level diagnostics, not an actor or research benchmark."""
import numpy as np
import mujoco
from rlexplore.config import Config
from rlexplore.physics import Physics
p=Physics(Config())
p.reset(np.random.default_rng(0), 'E0')
# Remove external geometry for isolated tracking diagnostics.
for name in ['table_0','table_1','block_0','block_1','occluder']:
    p.model.body_pos[p.model.body(name).id,0]=4
p.run(p.reference,1)
print('start',p.data.site_xpos[p.site].round(3),flush=True)
for target in [[.30,.26,.90],[.32,.26,.80],[.36,.26,.75],[.40,.28,.75]]:
    q,err=p.ik(target,np.eye(3))
    print('IK',target,err,'arm',q[15:22].round(3),flush=True)
    p.run(q,2.5)
    print('actual',p.data.site_xpos[p.site].round(3),'delta',(p.data.qpos[p.qids]-q)[12:22].round(3),flush=True)
    contacts=[]
    for c in p.data.contact:
        if c.dist<-.001:
            contacts.append(([p.model.body(p.model.geom_bodyid[g]).name for g in [c.geom1,c.geom2]],round(c.dist,3)))
    print('contacts',contacts,flush=True)
p.close()
