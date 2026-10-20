import unittest
import _path  # noqa: F401
from logad.parsing.ssh_parser import SSHParser

ONE_ATTEMPT = """Dec 10 06:55:46 LabSZ sshd[24200]: reverse mapping checking getaddrinfo for x.com [173.234.31.186] failed - POSSIBLE BREAK-IN ATTEMPT!
Dec 10 06:55:46 LabSZ sshd[24200]: Invalid user webmaster from 173.234.31.186
Dec 10 06:55:46 LabSZ sshd[24200]: input_userauth_request: invalid user webmaster [preauth]
Dec 10 06:55:46 LabSZ sshd[24200]: pam_unix(sshd:auth): check pass; user unknown
Dec 10 06:55:46 LabSZ sshd[24200]: pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=173.234.31.186 
Dec 10 06:55:48 LabSZ sshd[24200]: Failed password for invalid user webmaster from 173.234.31.186 port 38926 ssh2
Dec 10 06:55:48 LabSZ sshd[24200]: Connection closed by 173.234.31.186 [preauth]""".splitlines()


def parse(lines, year=2025, **kw):
    p = SSHParser(year, **kw).parse_lines(lines)
    return p.frame(), p.report


class ParserTests(unittest.TestCase):
    def test_one_failed_attempt_counts_once(self):
        df, rep = parse(ONE_ATTEMPT)
        self.assertEqual(int(df.loc[df.event_type == "LOGIN_FAILED", "repeat"].sum()), 1)
        self.assertEqual(sum(rep.ignored_redundant.values()), 2)   # input_userauth + check pass
        self.assertEqual(rep.event_lines, 5)

    def test_failure_method_rules(self):
        lines = [
            "Mar  1 10:00:00 h sshd[1]: Failed password for bob from 1.2.3.4 port 22 ssh2",
            "Mar  1 10:00:01 h sshd[1]: Failed none for invalid user x from 1.2.3.4 port 22 ssh2",
            "Mar  1 10:00:02 h sshd[1]: Failed publickey for bob from 1.2.3.4 port 22 ssh2",
            "Mar  1 10:00:03 h sshd[1]: Failed keyboard-interactive/pam for bob from 1.2.3.4 port 22 ssh2",
        ]
        df, _ = parse(lines)
        self.assertEqual(list(df.event_type),
                         ["LOGIN_FAILED", "AUTH_PROBE", "AUTH_FAILED_OTHER", "LOGIN_FAILED"])

    def test_message_repeated_sets_repeat(self):
        df, _ = parse(["Dec 10 07:00:00 h sshd[1]: message repeated 5 times: "
                       "[ Failed password for root from 112.95.230.3 port 46210 ssh2]"])
        self.assertEqual((df.event_type[0], df.username[0], int(df.repeat[0])), ("LOGIN_FAILED", "root", 5))

    def test_modern_formats(self):
        lines = [
            "Mar  1 10:00:00 h sshd[1]: Invalid user admin from 5.6.7.8 port 4444",
            "Mar  1 10:00:01 h sshd[1]: Connection closed by authenticating user root 5.6.7.8 port 4444 [preauth]",
            "Mar  1 10:00:02 h sshd[1]: Disconnected from invalid user admin 5.6.7.8 port 4444 [preauth]",
            "Mar  1 10:00:03 h sshd[1]: error: maximum authentication attempts exceeded for root from 5.6.7.8 port 4444 ssh2 [preauth]",
            "Mar  1 10:00:04 h sshd[1]: Accepted publickey for alice from 9.9.9.9 port 1 ssh2: RSA SHA256:abc",
            "Mar  1 10:00:05 h sshd-session[2]: Failed password for alice from 9.9.9.9 port 2 ssh2",
        ]
        df, rep = parse(lines)
        self.assertEqual(list(df.event_type), ["INVALID_USER", "CONNECTION_CLOSED", "DISCONNECT",
                                               "MAX_AUTH_EXCEEDED", "LOGIN_SUCCESS", "LOGIN_FAILED"])
        self.assertEqual(df.username[1], "root")
        self.assertEqual(rep.unsupported_sshd_lines, 0)

    def test_pam_failure_extracts_rhost_and_user(self):
        df, _ = parse(["Dec 10 07:01:00 h sshd(pam_unix)[24200]: authentication failure; logname= uid=0 "
                       "euid=0 tty=ssh ruser= rhost=187.141.143.180  user=root"])
        self.assertEqual((df.event_type[0], df.src_ip[0], df.username[0]),
                         ("PAM_AUTH_FAILURE", "187.141.143.180", "root"))

    def test_report_counts_everything(self):
        df, rep = parse([
            "Mar  1 10:00:00 h CRON[1]: pam_unix(cron:session): session opened for user root",
            "garbage line", "", "Mar  1 10:00:01 h sshd[1]: Something unseen 42",
            "Feb 30 10:00:00 h sshd[1]: Accepted password for a from 1.1.1.1 port 1 ssh2",
        ])
        r = rep.to_dict()
        self.assertEqual((r["not_sshd_lines"], r["malformed_lines"], r["blank_lines"],
                          r["unsupported_sshd_lines"], r["invalid_timestamp_lines"]), (1, 1, 1, 1, 1))
        self.assertTrue(df.empty)

    def test_year_rollover(self):
        df, rep = parse(["Dec 31 23:59:58 h sshd[1]: Accepted password for a from 1.1.1.1 port 1 ssh2",
                         "Jan  1 00:00:02 h sshd[1]: Accepted password for a from 1.1.1.1 port 1 ssh2"], year=2025)
        self.assertEqual([t.year for t in df.ts], [2025, 2026])
        self.assertEqual(rep.year_rollovers, 1)
        self.assertEqual((df.ts[1] - df.ts[0]).total_seconds(), 4)

    def test_late_december_line_after_rollover_keeps_old_year(self):
        df, rep = parse(["Dec 31 23:59:58 h sshd[1]: Accepted password for a from 1.1.1.1 port 1 ssh2",
                         "Jan  1 00:00:02 h sshd[1]: Accepted password for a from 1.1.1.1 port 1 ssh2",
                         "Dec 31 23:59:59 h sshd[1]: Accepted password for b from 1.1.1.1 port 1 ssh2"])
        self.assertEqual(rep.year_rollovers, 1)
        self.assertEqual(sorted(t.year for t in df.ts), [2025, 2025, 2026])

    def test_timezone_offset_to_utc(self):
        df, _ = parse(["Mar  1 10:00:00 h sshd[1]: Accepted password for a from 1.1.1.1 port 1 ssh2"],
                      tz_offset_hours=5.5)
        self.assertEqual((df.ts[0].hour, df.ts[0].minute), (4, 30))


if __name__ == "__main__":
    unittest.main()
