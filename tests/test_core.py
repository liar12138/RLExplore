import unittest
from dataclasses import replace
import numpy as np
import torch
from rlexplore.config import Config
from rlexplore.env import STATE_DIM, PRIV_DIM, VIEWS
from rlexplore.history import History, batch_histories
from rlexplore.memory import EvidenceMap
from rlexplore.model import ActorCritic
from rlexplore.ppo import returns_gae
from rlexplore.sonic import POLICY_ORDER, DEFAULT, FrozenSonic


def frame(cfg):
    rng = np.random.default_rng(0)
    return {"image": rng.random((4, 48, 64)).astype(np.float32),
            "state": rng.normal(size=STATE_DIM).astype(np.float32),
            "candidates": rng.normal(size=(cfg.max_candidates, 14)).astype(np.float32),
            "action_mask": np.r_[np.ones(len(VIEWS), bool), np.zeros(cfg.max_candidates, bool), True]}


class SequenceTests(unittest.TestCase):
    def test_all_backbones_gradients_and_privileged_actor_boundary(self):
        torch.set_num_threads(2)
        for name in ("reactive", "gru", "transformer", "ttt"):
            with self.subTest(name=name):
                cfg = Config(hidden_dim=32, heads=4, layers=1, backbone=name)
                net = ActorCritic(cfg)
                histories = [[frame(cfg)], [frame(cfg), frame(cfg), frame(cfg)]]
                inputs, valid = batch_histories(histories, "cpu")
                dist, value = net(inputs, valid, torch.zeros(2, PRIV_DIM))
                other, _ = net(inputs, valid, torch.ones(2, PRIV_DIM)*100)
                torch.testing.assert_close(dist.logits, other.logits)
                self.assertTrue(torch.isfinite(dist.logits).all())
                self.assertEqual(float(dist.probs[:, len(VIEWS):-1].sum().detach()), 0.)
                loss = -dist.log_prob(torch.tensor([0, 2])).mean()+value.square().mean()
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
                # The same parameters/context must produce PPO ratio exactly one.
                new, _ = net(inputs, valid)
                torch.testing.assert_close(new.log_prob(torch.tensor([0, 2])), dist.log_prob(torch.tensor([0, 2])))

    def test_padding_does_not_change_policy(self):
        for name in ("gru", "transformer", "ttt"):
            cfg = Config(backbone=name, hidden_dim=32, layers=1)
            net = ActorCritic(cfg).eval()
            short, long = [frame(cfg)], [frame(cfg)]*4
            x, v = batch_histories([short], "cpu")
            a, _ = net(x, v)
            x, v = batch_histories([short, long], "cpu")
            b, _ = net(x, v)
            torch.testing.assert_close(a.probs[0], b.probs[0], atol=1e-6, rtol=1e-6)

    def test_episode_history_reset(self):
        cfg = Config()
        h = History(2)
        h.reset(frame(cfg))
        h.append(frame(cfg))
        saved = h.snapshot()
        h.reset(frame(cfg))
        self.assertEqual(len(h.snapshot()), 1)
        self.assertEqual(len(saved), 2)

    def test_gae_terminal_and_timeout(self):
        a, r = returns_gae([1, 2], [0, 0], [5, 100], [False, True], [True, True], [2, 1], .9, .95)
        self.assertAlmostEqual(float(r[0]), 1+.9**2*5, places=5)
        self.assertEqual(float(r[1]), 2.)
        # Reset boundaries must not propagate the next episode's reward.
        b, _ = returns_gae([1, 200], [0, 0], [5, 100], [False, True], [True, True], [2, 1], .9, .95)
        self.assertEqual(float(a[0]), float(b[0]))

    def test_vision_does_not_mark_occluded_cells_free(self):
        memory = EvidenceMap()
        self.assertTrue(np.all(memory.occupancy == -1))
        camera = np.array([.1, 0, 1.2])
        depth = np.full((48, 64), .4)
        memory.integrate(depth, camera, np.eye(3), 60)
        surface = camera + [0, 0, -.4]
        behind = surface + [0, 0, -.15]
        index, valid = memory.indices(behind[None])
        self.assertTrue(valid[0])
        self.assertEqual(int(memory.occupancy[tuple(index[0])]), -1)
        before = memory.occupancy.copy()
        # Idempotent same rays may add evidence, but cannot reveal hidden space.
        memory.integrate(depth, camera, np.eye(3), 60)
        np.testing.assert_array_equal(memory.occupancy, before)

    def test_joint_permutation_bijection(self):
        np.testing.assert_array_equal(np.sort(POLICY_ORDER), np.arange(29))
        action = np.arange(29.)
        mj_action = np.empty(29)
        mj_action[POLICY_ORDER] = action
        np.testing.assert_array_equal(mj_action[POLICY_ORDER], action)


class RuntimeTests(unittest.TestCase):
    def test_real_sonic_inference_and_reset(self):
        cfg = Config()
        sonic = FrozenSonic(cfg.sonic_dir)
        output = sonic.command(DEFAULT, np.zeros(29), np.zeros(3), np.eye(3), DEFAULT)
        self.assertEqual(output.shape, (29,))
        self.assertTrue(np.isfinite(output).all())
        self.assertEqual(sonic.calls, 1)
        sonic.reset()
        repeat = sonic.command(DEFAULT, np.zeros(29), np.zeros(3), np.eye(3), DEFAULT)
        np.testing.assert_allclose(output, repeat, atol=1e-6)

    def test_sonic_g1_mode_is_scalar_zero_and_references_are_policy_order(self):
        sonic = FrozenSonic(Config().sonic_dir)
        real_encoder = sonic.encoder
        captured = []
        class Capture:
            def get_inputs(self):
                return real_encoder.get_inputs()
            def run(self, names, feed):
                captured.append(next(iter(feed.values())).copy())
                return real_encoder.run(names, feed)
        sonic.encoder = Capture()
        reference = DEFAULT + np.linspace(0, .1, 29)
        sonic.command(DEFAULT, np.zeros(29), np.zeros(3), np.eye(3), reference)
        np.testing.assert_array_equal(captured[0][0, :4], 0)
        np.testing.assert_allclose(captured[0][0, 4:294].reshape(10, 29),
                                   np.tile(reference[POLICY_ORDER], (10, 1)), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
