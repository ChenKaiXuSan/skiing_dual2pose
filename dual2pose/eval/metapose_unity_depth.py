"""Streaming train-only input-consistency guards; never a GT accuracy estimate."""
import numpy as np

POLICY = dict(min_reference_depth_fraction=.01, min_depth_cosine=0.,
              min_depth_ratio=.1, max_depth_ratio=10., max_bad_fraction=.01,
              min_radius_ratio=.1, max_radius_ratio=10., min_camera_scale=1e-4)


class DepthHealth:
    def __init__(self):
        self.frames = self.informative = self.bad_depth = self.bad_geometry = 0
        self.min_scale = float('inf')
        self.cosine_sum = 0.

    def update(self, projected, reference, scales):
        p,r,s=(np.asarray(a,dtype=np.float64) for a in (projected,reference,scales))
        if p.shape != r.shape or p.ndim != 4 or p.shape[1:] != (2,13,3) or s.shape != p.shape[:2] or not len(p):
            raise ValueError('Require nonempty (N,2,13,3) projections/reference and (N,2) scales')
        if not all(np.isfinite(v).all() for v in (p,r,s)) or np.any(s<=0):
            raise ValueError('Nonfinite pose or nonpositive scale')
        pc=p-p.mean(-2,keepdims=True);rc=r-r.mean(-2,keepdims=True)
        radius=np.linalg.norm(pc,axis=-1).mean(-1);base=np.linalg.norm(rc,axis=-1).mean(-1)
        if np.any(base<=1e-8):raise ValueError('Degenerate S1 reference')
        ratio=radius/base
        self.bad_geometry+=int(((ratio<POLICY['min_radius_ratio'])|(ratio>POLICY['max_radius_ratio'])).any(-1).sum())
        rz=rc[...,2];pz=pc[...,2];rn=np.linalg.norm(rz,axis=-1);pn=np.linalg.norm(pz,axis=-1)
        info=rn>POLICY['min_reference_depth_fraction']*np.linalg.norm(rc,axis=(-2,-1))
        cosine=np.divide(np.sum(rz*pz,axis=-1),rn*pn,out=np.zeros_like(rn),where=rn*pn>1e-15)
        dr=np.divide(pn,rn,out=np.zeros_like(rn),where=rn>1e-15)
        bad=(cosine<=POLICY['min_depth_cosine'])|(dr<POLICY['min_depth_ratio'])|(dr>POLICY['max_depth_ratio'])
        self.informative+=int(info.sum());self.bad_depth+=int((bad&info).sum())
        self.cosine_sum+=float(cosine[info].sum());self.min_scale=min(self.min_scale,float(s.min()));self.frames+=len(p)

    def result(self):
        if not self.frames:raise ValueError('Empty health coverage')
        df=self.bad_depth/self.informative if self.informative else 1.
        gf=self.bad_geometry/self.frames
        reasons=[]
        if not self.informative:reasons.append('no_informative_reference_depth')
        if df>POLICY['max_bad_fraction']:reasons.append('depth_direction_or_collapse')
        if gf>POLICY['max_bad_fraction']:reasons.append('geometry_radius')
        if self.min_scale<POLICY['min_camera_scale']:reasons.append('camera_scale')
        return dict(passed=not reasons,reasons=reasons,frames=self.frames,
                    informative_view_frames=self.informative,depth_bad_view_frames=self.bad_depth,
                    depth_bad_fraction=df,geometry_bad_frames=self.bad_geometry,geometry_bad_fraction=gf,
                    mean_depth_cosine=self.cosine_sum/self.informative if self.informative else None,
                    min_camera_scale=self.min_scale,policy=dict(POLICY))
