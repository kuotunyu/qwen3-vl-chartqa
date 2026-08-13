from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github" / "scripts" / "check_boundary.py"
)


def load_checker():
    spec = importlib.util.spec_from_file_location("check_boundary_under_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicationBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        subprocess.run(["git", "init", "-b", "main", self.repo], check=True)
        subprocess.run(
            ["git", "-C", self.repo, "config", "user.name", "kuotunyu"], check=True
        )
        subprocess.run(
            [
                "git",
                "-C",
                self.repo,
                "config",
                "user.email",
                "61350295+kuotunyu@users.noreply.github.com",
            ],
            check=True,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def commit(self, relative_path: str, content: str) -> None:
        target = self.repo / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        subprocess.run(["git", "-C", self.repo, "add", relative_path], check=True)
        subprocess.run(
            ["git", "-C", self.repo, "commit", "-m", "test fixture"], check=True
        )

    def run_checker(self) -> tuple[int, str]:
        checker = load_checker()
        stdout = io.StringIO()
        with (
            patch.object(checker, "ROOT", self.repo),
            contextlib.redirect_stdout(stdout),
        ):
            return checker.main(), stdout.getvalue()

    def test_rejects_embedded_chartqa_case_answer(self) -> None:
        leak = "expected answer `" + "96`"
        self.commit("public.md", f"A raw ChartQA row returned the {leak}.\n")

        status, output = self.run_checker()

        self.assertEqual(1, status)
        self.assertIn("ChartQA case answer", output)

    def test_rejects_hash_anonymity_claim(self) -> None:
        claim = "one-way `query_" + "sha256`"
        self.commit(
            "public.md", f"Evidence uses a {claim} that does not reveal the query.\n"
        )

        status, output = self.run_checker()

        self.assertEqual(1, status)
        self.assertIn("hash anonymity claim", output)

    def test_allows_identity_like_hex_sequence_inside_hash(self) -> None:
        digest = "674a" + "12383e3c38a1bcccae7d4f3633b37852230b6047883cb2f4c2d1b36d9bf5"
        self.commit("public.md", f"sha256:{digest}\n")

        status, output = self.run_checker()

        self.assertEqual(0, status, output)

    def test_rejects_standalone_superseded_identity(self) -> None:
        identity = "A" + "123"
        self.commit("public.md", f"Former test account: {identity}\n")

        status, output = self.run_checker()

        self.assertEqual(1, status)
        self.assertIn("superseded private identity", output)


if __name__ == "__main__":
    unittest.main()
