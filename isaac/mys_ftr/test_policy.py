"""Portable actor equivalence and FT-R frame invariance; no Isaac installation needed."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from policy import Actor, Footholds, quat, rotation

ROOT = Path(__file__).resolve().parent


class FrozenPolicyTests(unittest.TestCase):
    def test_matches_original_jax_actor(self):
        with np.load(ROOT / 'actor.npz') as saved:
            actor = Actor(dict(saved))
        with np.load(ROOT / 'jax_reference.npz') as saved:
            np.testing.assert_allclose(actor(saved['actor_observation']), saved['action'], atol=2e-5, rtol=2e-5)

    def test_bundled_actor_integrity(self):
        meta = json.loads((ROOT / 'export.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT / 'actor.npz').read_bytes()).hexdigest(), meta['actor_sha256'])
        self.assertEqual(meta['actor_dims'], [91, 512, 256, 128, 23])

    def test_global_frame_change_preserves_root_relative_observation(self):
        root_pos = np.array([1.2, -.3, .66])
        root_quat = quat(Rotation.from_euler('xyz', [.03, -.05, .2]))
        foot_pos = np.array([[1.15, -.19, 0.], [1.16, -.41, 0.]])
        foot_quat = np.array([quat(Rotation.from_euler('z', .18)), quat(Rotation.from_euler('z', .22))])
        omega = np.array([.2, -.1, .3])
        q, dq = np.linspace(-.2, .3, 23), np.linspace(-.5, .5, 23)
        command = np.array([.25, 0., 0., 1.])
        base = Footholds(foot_pos, foot_quat)
        first = base.observation(root_pos, root_quat, q, dq, omega, foot_pos, foot_quat, command, 0.)
        change = Rotation.from_euler('z', 1.1)
        shift = np.array([7., -4., 0.])
        changed_pos = change.apply(foot_pos) + shift
        changed_quat = np.array([quat(change * rotation(x)) for x in foot_quat])
        other = Footholds(changed_pos, changed_quat)
        second = other.observation(change.apply(root_pos) + shift, quat(change * rotation(root_quat)),
                                   q, dq, change.apply(omega), changed_pos, changed_quat, command, 0.)
        np.testing.assert_allclose(first, second, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
