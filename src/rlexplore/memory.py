"""Ray-consistent evidence; no simulator IDs, segmentation or hidden objects."""
import numpy as np


class EvidenceMap:
    def __init__(self, resolution=.025):
        self.resolution = resolution
        self.origin = np.array([0., -.7, .4])
        self.shape = np.array([40, 56, 40])
        self.reset()

    def reset(self):
        self.occupancy = np.full(tuple(self.shape), -1, np.int8)  # unknown/free/occupied
        self.support = np.zeros(tuple(self.shape), np.int16)
        self.last_seen = np.zeros(tuple(self.shape), np.int32)
        self.updates = 0

    def indices(self, points):
        ids = np.floor((points - self.origin) / self.resolution).astype(int)
        valid = np.all((ids >= 0) & (ids < self.shape), axis=-1)
        return ids, valid

    def integrate(self, depth, position, rotation, fovy):
        h, w = depth.shape
        f = h / (2 * np.tan(np.deg2rad(fovy) / 2))
        yy, xx = np.mgrid[:h, :w]
        local = np.stack([(xx + .5-w/2)*depth/f, -(yy+.5-h/2)*depth/f, -depth], axis=-1)
        world = local @ rotation.T + position
        dy, dx = np.gradient(world, axis=(0, 1))
        normals = np.cross(dx, dy)
        nz = np.abs(normals[..., 2]) / np.maximum(np.linalg.norm(normals, axis=-1), 1e-8)
        good = np.isfinite(depth) & (depth > .05) & (depth < 2.5)
        good[::2, ::2] &= True
        sampled = np.zeros_like(good)
        sampled[::2, ::2] = True
        good &= sampled
        points = world[good]
        self.updates += 1
        # Only cells strictly before measured surfaces become free.
        for fraction in np.linspace(.03, .97, 40):
            ids, valid = self.indices(position + fraction * (points-position))
            q = ids[valid]
            self.occupancy[tuple(q.T)] = 0
        ids, valid = self.indices(points)
        q = ids[valid]
        self.occupancy[tuple(q.T)] = 1
        self.last_seen[tuple(q.T)] = self.updates
        horizontal = (nz[good] > .94)[valid]
        # Avoid depth discontinuity edges: small surface-gradient only.
        smooth = (np.maximum(np.linalg.norm(dx, axis=-1), np.linalg.norm(dy, axis=-1))[good] < .04)[valid]
        flat = q[horizontal & smooth]
        np.add.at(self.support, tuple(flat.T), 1)

    def candidates(self, halfsize, maximum=16):
        ids = np.argwhere((self.support >= 2) & (self.occupancy == 1))
        if not len(ids):
            return np.zeros((maximum, 14), np.float32), np.zeros(maximum, bool)
        points = self.origin + (ids+.5)*self.resolution
        # Fixed geometric order/spacing; no hidden feasibility selection.
        order = np.lexsort((points[:, 1], points[:, 0], -points[:, 2]))
        chosen = []
        for k in order:
            p = points[k]
            if not (.48 < p[2] < 1.05 and .20 < p[0] < .85):
                continue
            if all(np.linalg.norm(p-q) > .065 for q in chosen):
                chosen.append(p)
            if len(chosen) == maximum:
                break
        features = np.zeros((maximum, 14), np.float32)
        valid = np.zeros(maximum, bool)
        for j, p in enumerate(chosen):
            footprint = np.array([[x, y, 0] for x in np.arange(-halfsize[0], halfsize[0]+.01, .025)
                                   for y in np.arange(-halfsize[1], halfsize[1]+.01, .025)])
            bottom, inbounds = self.indices(p + footprint)
            known_support = np.zeros(len(bottom), bool)
            qb = bottom[inbounds]
            known_support[inbounds] = self.support[tuple(qb.T)] > 0
            clearance = p + footprint[:, None, :] + np.array([0, 0, 1]) * np.arange(.04, .2, .025)[None, :, None]
            qi, ok = self.indices(clearance.reshape(-1, 3))
            states = np.full(len(qi), -1, int)
            states[ok] = self.occupancy[tuple(qi[ok].T)]
            at, _ = self.indices(p[None])
            age = self.updates-self.last_seen[tuple(at[0])]
            features[j] = np.r_[p, np.linalg.norm(p[:2]), known_support.mean(),
                                 np.mean(states == 0), np.mean(states == 1), np.mean(states == -1),
                                 age/32, halfsize*2, 0., 1.]
            valid[j] = True
        return features, valid

    def summary(self):
        return np.array([(self.occupancy == 0).mean(), (self.occupancy == 1).mean(),
                         (self.occupancy == -1).mean(), (self.support > 0).mean()], np.float32)
