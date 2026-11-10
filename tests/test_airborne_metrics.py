"""Force sampling distinguishes flight intervals from mere clip completion."""

import numpy as np

from supermimic.deployment.metrics import airborne_metrics


def test_airborne_runs_reject_contact_and_short_gaps():
    contacts = np.zeros((28, 13))
    contacts[:, 3] = 100  # left vertical force
    contacts[2:10, 3] = 0  # 40 ms of both feet unloaded
    contacts[12:15, 3] = 0  # only 15 ms
    contacts[18:28, 3] = 0
    contacts[18:28, 10] = 1  # hand/other ground support is not flight
    stats = airborne_metrics(contacts, 0.005)
    assert stats["airborne_substeps"] == 11
    assert stats["longest_airborne_s"] == 0.04
    assert stats["airborne_intervals_ge_40ms"] == 1
    contacts[:, 6] = 100  # right foot alone supports the body
    assert airborne_metrics(contacts, 0.005)["airborne_substeps"] == 0
    assert airborne_metrics(np.zeros((0, 13)), 0.005)["longest_airborne_s"] == 0
