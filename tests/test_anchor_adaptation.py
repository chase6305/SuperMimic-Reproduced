"""A feedback warm start must preserve behavior, optimizer state and provenance."""

from dataclasses import asdict

import pytest
import torch

from supermimic.artifacts import sha256
from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg
from supermimic.training.anchor import enable_anchor_feedback


def test_anchor_migration_preserves_values_moments_and_learns_new_inputs(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    pcfg = PolicyCfg(encoder_hidden=(16, 8), decoder_hidden=(16, 8))
    ccfg = CriticCfg(hidden=(16, 8))
    actor, critic = Policy(pcfg), Critic(ccfg, pcfg=pcfg)
    actor_opt, critic_opt = (
        torch.optim.Adam(actor.parameters()),
        torch.optim.Adam(critic.parameters()),
    )
    obs, goal, feat, xi, error = (torch.randn(3, n) for n in (930, 680, 71, 33, 4))
    actor(obs, goal).square().mean().backward()
    actor_opt.step()
    critic(feat, goal, xi, error).square().mean().backward()
    critic_opt.step()
    source, output = tmp_path / "source.pt", tmp_path / "anchor.pt"
    original = {
        "iteration": 123,
        "policy": actor.state_dict(),
        "critic": critic.state_dict(),
        "policy_opt": actor_opt.state_dict(),
        "critic_opt": critic_opt.state_dict(),
        "wm": {"unchanged": torch.randn(4, 5)},
        "torch_rng": torch.get_rng_state(),
        "config": {
            "env": {"future_frames": 10, "anchor_obs": False},
            "policy": asdict(pcfg),
            "critic": asdict(ccfg),
            "policy_trainer": {},
        },
    }
    torch.save(original, source)
    source_hash = sha256(source)
    report = enable_anchor_feedback(source, output)
    ck = torch.load(output, weights_only=False)
    assert sha256(source) == source_hash == report["source_checkpoint_sha256"]
    assert report["checkpoint_sha256"] == sha256(output)
    assert ck["config"]["env"]["anchor_obs"] and ck["iteration"] == 123
    torch.testing.assert_close(ck["wm"]["unchanged"], original["wm"]["unchanged"])
    torch.testing.assert_close(ck["torch_rng"], original["torch_rng"])
    pcfg.goal_dim = 689
    adapted_actor, adapted_critic = Policy(pcfg), Critic(ccfg, pcfg=pcfg)
    adapted_actor.load_state_dict(ck["policy"])
    adapted_critic.load_state_dict(ck["critic"])
    extended = torch.cat([goal, torch.randn(3, 9)], -1)
    torch.testing.assert_close(adapted_actor(obs, extended), actor(obs, goal))
    torch.testing.assert_close(
        adapted_critic(feat, extended, xi, error), critic(feat, goal, xi, error)
    )
    for before, after, model, offset in (
        (original["policy_opt"], ck["policy_opt"], adapted_actor, 680),
        (original["critic_opt"], ck["critic_opt"], adapted_critic, 751),
    ):
        new_opt = torch.optim.Adam(model.parameters())
        new_opt.load_state_dict(after)
        for old_id, new_id in zip(before["state"], after["state"], strict=True):
            for key in ("step", "exp_avg", "exp_avg_sq"):
                a, b = before["state"][old_id][key], after["state"][new_id][key]
                if a.shape != b.shape:
                    torch.testing.assert_close(
                        b[:, offset : offset + 9], torch.zeros_like(b[:, offset : offset + 9])
                    )
                    b = torch.cat([b[:, :offset], b[:, offset + 9 :]], 1)
                torch.testing.assert_close(a, b)
        new_opt.zero_grad()
        loss = (
            model(obs, extended).sum()
            if model is adapted_actor
            else model(feat, extended, xi, error).sum()
        )
        loss.backward()
        first = next(model.parameters())
        assert first.grad[:, offset : offset + 9].abs().sum() > 0
        new_opt.step()  # migrated Adam tensors must also support a real update
        assert first[:, offset : offset + 9].abs().sum() > 0
    with pytest.raises(FileExistsError):
        enable_anchor_feedback(source, output)
    with pytest.raises(ValueError, match="already"):
        enable_anchor_feedback(output, tmp_path / "again.pt")
