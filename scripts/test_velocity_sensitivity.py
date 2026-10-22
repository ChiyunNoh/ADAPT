"""Regression checks for shared velocity timing, frames, and reset isolation."""

from types import SimpleNamespace
import unittest

import torch

from scripts.evaluate_velocity_sensitivity import (
    VelocityInput, VelocitySensitivityMixin, velocity_observation,
)
from mjlab.utils.lab_api.math import quat_apply_inverse


class VelocityMeasurementTests(unittest.TestCase):
    def test_delay_and_partial_reset_do_not_leak_or_advance(self):
        channel = VelocityInput([0, 0], [0, 2], 1, 42, "cpu")
        reset = torch.zeros(2, dtype=torch.bool)
        for value in range(4):
            measured = channel.push(torch.full((2, 3), float(value)), reset)
            torch.testing.assert_close(measured[0], torch.full((3,), float(value)))
            torch.testing.assert_close(measured[1], torch.full((3,), float(max(0, value - 2))))
        index = channel.index
        untouched = channel.buffer[:, 0].clone()
        channel.reset(torch.full((2, 3), 99.), torch.tensor([1]))
        self.assertEqual(channel.index, index)
        torch.testing.assert_close(channel.buffer[:, 0], untouched)
        torch.testing.assert_close(channel.last[0], torch.full((3,), 3.))
        torch.testing.assert_close(channel.last[1], torch.full((3,), 99.))

    def test_paired_noise_and_input_immutability(self):
        channel = VelocityInput([.05, .5], [0, 0], 2, 42, "cpu")
        clean = torch.zeros(4, 3)
        measured = channel.push(clean, torch.zeros(4, dtype=torch.bool))
        torch.testing.assert_close(measured[:2] * 10, measured[2:])
        self.assertLessEqual(measured[:2].abs().max().item(), .05)
        self.assertTrue(torch.equal(clean, torch.zeros_like(clean)))

    def make_env(self, target="shared", amplitude=.2, lag=0):
        env = VelocitySensitivityMixin()
        env.device, env.num_envs = "cpu", 2
        params = dict(sensor_name="unused", amplitudes=[amplitude], lags=[lag],
                      trials=2, seed=42, target=target)
        env.cfg = SimpleNamespace(observations={"actor": SimpleNamespace(
            terms={"base_lin_vel": SimpleNamespace(params=params)})})
        env.velocity_sensitivity_input = VelocityInput([amplitude], [lag], 2, 42, "cpu")
        env.velocity_sensor_offset = torch.tensor([[.04525, 0, -.08339]]).expand(2, 3)
        env.velocity_clean_sample = torch.zeros(2, 3)
        # Nontrivial orientations and angular velocity expose frame/lever-arm errors.
        qpos = torch.zeros(2, 7)
        qpos[:, 3:7] = torch.tensor([[.70710678, 0, 0, .70710678], [.5, .5, .5, .5]])
        qvel = torch.tensor([[1., 2., 3., .2, .3, .4], [-1., .5, 2., -.4, .6, -.8]])
        env.sim = SimpleNamespace(data=SimpleNamespace(qpos=qpos, qvel=qvel))
        env.ideal_qvel_view = qvel.clone()
        return env, params

    def test_shared_error_frames_lever_arm_and_observation_reads(self):
        env, params = self.make_env()
        true_qvel = env.sim.data.qvel.clone()
        env._prepare_observer_inputs(torch.arange(2))
        quat = env.sim.data.qpos[:, 3:7]
        sensor_from_observer = quat_apply_inverse(quat, env.ideal_qvel_view[:, :3]) + torch.cross(
            env.ideal_qvel_view[:, 3:6], env.velocity_sensor_offset, dim=-1)
        torch.testing.assert_close(sensor_from_observer, env.velocity_sensitivity_input.last)
        torch.testing.assert_close(env.sim.data.qvel, true_qvel)
        index = env.velocity_sensitivity_input.index
        rng = env.velocity_sensitivity_input.generator.get_state().clone()
        first = velocity_observation(env, **params)
        second = velocity_observation(env, **params)
        torch.testing.assert_close(first, second)
        first[:] = 99
        self.assertFalse(torch.equal(first, env.velocity_sensitivity_input.last))
        self.assertEqual(index, env.velocity_sensitivity_input.index)
        self.assertTrue(torch.equal(rng, env.velocity_sensitivity_input.generator.get_state()))

    def test_zero_error_and_actor_control_leave_observer_unchanged(self):
        for target, amplitude in [("shared", 0), ("actor", .5)]:
            env, _ = self.make_env(target, amplitude)
            before = env.ideal_qvel_view.clone()
            env._prepare_observer_inputs(torch.arange(2))
            self.assertTrue(torch.equal(before, env.ideal_qvel_view))

    def test_delayed_sample_shared_before_observer_dynamics(self):
        env, _ = self.make_env(amplitude=0, lag=1)
        env._prepare_observer_inputs(torch.arange(2))
        initial = env.velocity_sensitivity_input.last.clone()
        env.sim.data.qvel[:, :3] += 1
        env.ideal_qvel_view[:] = env.sim.data.qvel
        env._prepare_observer_inputs()
        torch.testing.assert_close(env.velocity_sensitivity_input.last, initial)
        quat = env.sim.data.qpos[:, 3:7]
        sensor_from_observer = quat_apply_inverse(quat, env.ideal_qvel_view[:, :3]) + torch.cross(
            env.ideal_qvel_view[:, 3:6], env.velocity_sensor_offset, dim=-1)
        torch.testing.assert_close(sensor_from_observer, initial)


if __name__ == "__main__":
    unittest.main()
