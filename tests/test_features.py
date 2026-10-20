import unittest
import numpy as np
import _path  # noqa: F401
from logad.features.baselines import add_baseline_features
from logad.features.extractors import (
    active_features, compute_raw_windows, max_weighted_burst)
from logad.features.split import chronological_split
from logad.features.windowing import assign_windows
from logad.parsing.normalizer import add_empty_labels
from logad.parsing.ssh_parser import SSHParser

KW = dict(group_by="user", tz_offset_hours=0, success_after_failed_min=5,
          off_hours_start=8, off_hours_end=19)


def raw_from(lines, group_by="user", blocklist=frozenset()):
    p = SSHParser(2025).parse_lines(lines)
    ev, stats = assign_windows(add_empty_labels(p.frame()), 300, group_by)
    kw = dict(KW, group_by=group_by)
    return compute_raw_windows(ev, blocklist_ips=blocklist, **kw), stats


def L(sec, msg, host="h1", minute=0):
    return f"Mar  3 10:{minute:02d}:{sec:02d} {host} sshd[1]: {msg}"


class FeatureTests(unittest.TestCase):
    def test_hand_computed_window(self):
        lines = [L(s, "Failed password for alice from 1.1.1.1 port 1 ssh2") for s in (0, 10, 20)]
        lines += [L(30, "Failed password for alice from 2.2.2.2 port 1 ssh2", host="h2")]
        lines += [L(50, "Failed password for alice from 1.1.1.1 port 1 ssh2")]
        lines += [L(59, "Failed password for alice from 1.1.1.1 port 1 ssh2")]
        lines += [L(5, "Accepted password for alice from 1.1.1.1 port 1 ssh2", minute=1)]
        raw, _ = raw_from(lines, blocklist=frozenset({"2.2.2.2"}))
        self.assertEqual(len(raw), 1)
        r = raw.iloc[0]
        self.assertEqual(r.failed_count, 6)
        self.assertAlmostEqual(r.failed_ratio, 6 / 7)
        self.assertEqual(r.max_failed_burst_60s, 6)            # all within [0, 59]
        self.assertEqual(r.consecutive_failed_max, 6)
        self.assertEqual(r.distinct_hosts_failed, 2)
        self.assertEqual(r.success_after_failed, 1)
        self.assertEqual(r.event_count, 7)
        self.assertEqual(r.distinct_src_ips, 2)
        self.assertEqual(r.max_events_per_ip, 6)
        self.assertEqual((r.blocklisted_ip_count, r.blocklisted_event_count), (1, 1))
        self.assertEqual(r.is_off_hours, 0)
        self.assertEqual(r.is_weekend, 0)                       # Mon 3 Mar 2025
        self.assertAlmostEqual(r.hour_sin, np.sin(2 * np.pi * 10 / 24))

    def test_consecutive_resets_on_success_and_saf_needs_five(self):
        lines = [L(1, "Failed password for bob from 1.1.1.1 port 1 ssh2"),
                 L(2, "Failed password for bob from 1.1.1.1 port 1 ssh2"),
                 L(3, "Accepted password for bob from 1.1.1.1 port 1 ssh2"),
                 L(4, "Failed password for bob from 1.1.1.1 port 1 ssh2")]
        r = raw_from(lines)[0].iloc[0]
        self.assertEqual((r.consecutive_failed_max, r.success_after_failed), (2, 0))

    def test_burst_uses_repeat_weights_and_half_open_span(self):
        t, w = np.array([0.0, 30.0, 60.0]), np.array([1, 1, 1])
        self.assertEqual(max_weighted_burst(t, w, 60), 2)       # [0,60) excludes t=60
        self.assertEqual(max_weighted_burst(np.array([5.0]), np.array([7]), 60), 7)

    def test_off_hours_and_weekend(self):
        lines = ["Mar  8 03:00:00 h sshd[1]: Accepted password for a from 1.1.1.1 port 1 ssh2"]  # Saturday
        r = raw_from(lines)[0].iloc[0]
        self.assertEqual((r.is_off_hours, r.is_weekend), (1, 1))

    def test_unattributed_events_are_counted_not_hidden(self):
        lines = [L(1, "Received disconnect from 3.3.3.3: 11: Bye Bye [preauth]"),
                 L(2, "Accepted password for a from 1.1.1.1 port 1 ssh2")]
        raw, stats = raw_from(lines)
        self.assertEqual(stats["unattributed_events"], 1)

    def test_src_ip_grouping_and_feature_selection(self):
        lines = [L(1, "Failed password for a from 1.1.1.1 port 1 ssh2"),
                 L(2, "Failed password for b from 1.1.1.1 port 1 ssh2")]
        raw, _ = raw_from(lines, group_by="src_ip")
        self.assertEqual((raw.iloc[0].entity, raw.iloc[0].distinct_users), ("1.1.1.1", 2))
        feats, dropped = active_features("src_ip", False, True)
        self.assertIn("distinct_users", feats)
        self.assertNotIn("distinct_src_ips", feats)
        feats_u, dropped_u = active_features("user", False, False)
        self.assertNotIn("geo_country_change", feats_u)
        self.assertNotIn("blocklisted_ip_count", feats_u)
        self.assertIn("distinct_users", dropped_u)

    def test_unlabeled_windows_are_unknown_not_normal(self):
        raw, _ = raw_from([L(1, "Accepted password for a from 1.1.1.1 port 1 ssh2")])
        self.assertTrue(np.isnan(raw.iloc[0].label))


class SplitTests(unittest.TestCase):
    def test_chronological_and_ties_stay_together(self):
        import pandas as pd
        ts = pd.Series(pd.to_datetime(
            ["2025-01-01"] * 4 + ["2025-01-02"] * 3 + ["2025-01-03"] * 2 + ["2025-01-04"], utc=True))
        split, cut = chronological_split(ts, 0.6, 0.2, 0.2)
        by_day = {d: set(split[ts == d]) for d in ts.unique()}
        self.assertTrue(all(len(v) == 1 for v in by_day.values()))
        order = {"train": 0, "val": 1, "test": 2}
        ranks = [order[s] for s in split[np.argsort(ts.to_numpy(), kind="stable")]]
        self.assertEqual(ranks, sorted(ranks))


if __name__ == "__main__":
    unittest.main()
