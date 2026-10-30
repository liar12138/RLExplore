import unittest
from unittest import mock
from dataclasses import replace
import numpy as np
from rlexplore.config import Config
from rlexplore.physics import Physics
from rlexplore.env import PlacementEnv


class PhysicalIntegrationTests(unittest.TestCase):
    @mock.patch("mujoco.mj_step", side_effect=AssertionError("mjlab must not step CPU physics"))
    def test_sonic_release_retract_and_support(self, cpu_step):
        p = Physics(Config())
        try:
            p.reset(np.random.default_rng(0), "E0")
            p.run(p.reference, 1.)
            target = p.model.body_pos[p.model.body("table_0").id] + [0, 0, .025]
            success, reason, released = p.place(target)
            self.assertTrue(released, reason)
            self.assertTrue(success, f"{reason}: {p.last_stability}")
            self.assertFalse(p.data.eq_active[p.hold])
            self.assertGreater(p.sonic.calls, 100)
            self.assertIsNotNone(p.runtime)
            self.assertEqual(p.runtime.steps, p.low_steps*4)
            self.assertTrue(p.runtime.sim.wp_device.is_cuda)
            if p.data.ncon:
                self.assertTrue(np.all(p.data.contact.geom >= 0))
                self.assertTrue(np.all(p.data.contact.geom < p.model.ngeom))
        finally:
            p.close()

    def test_observation_contract_and_matched_object_layout(self):
        cfg = Config(image_width=64, image_height=48, max_candidates=8)
        env = PlacementEnv(cfg)
        try:
            obs, _ = env.reset(seed=3)
            self.assertTrue(env.observation_space.contains(obs))
            obs, reward, terminated, truncated, info = env.step(1)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertTrue(np.isfinite(reward))
            self.assertGreater(info["duration"], 0)
            env.step(env.n_actions-1)
            with self.assertRaises(RuntimeError):
                env.step(0)
        finally:
            env.close()
        p = Physics(replace(cfg, object_variant="small"))
        try:
            p.reset(np.random.default_rng(42), "E3")
            small_layout = p.model.body_pos.copy()
            p.cfg = replace(cfg, object_variant="large")
            p.reset(np.random.default_rng(42), "E3")
            np.testing.assert_array_equal(small_layout, p.model.body_pos)
            bound = p.model.geom_rbound[p.model.geom("payload_geom").id]
            self.assertAlmostEqual(bound, np.linalg.norm(p.halfsize), places=6)
        finally:
            p.close()


if __name__ == "__main__":
    unittest.main()
