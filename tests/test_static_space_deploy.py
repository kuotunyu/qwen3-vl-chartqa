from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy_hf_static_space.py"


def load_deployer():
    spec = importlib.util.spec_from_file_location("static_space_deployer", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StaticSpacePreflightTests(unittest.TestCase):
    def test_requires_the_synthetic_public_case_asset(self) -> None:
        deployer = load_deployer()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            for relative in deployer.REQUIRED_FILES:
                target = source / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("placeholder", encoding="utf-8")
            (source / "README.md").write_text(
                "sdk: static\napp_file: index.html\ncolorFrom: red\n", encoding="utf-8"
            )
            (source / "index.html").write_text(
                '<html lang="zh-Hant">demo_gradio_colab.ipynb '
                "5e5860f5d406 85.52%</html>",
                encoding="utf-8",
            )

            with patch.object(deployer, "SPACE_SOURCE", source):
                deployer.preflight()

            (source / "assets" / "synthetic_ood_04.png").unlink(missing_ok=True)
            with patch.object(deployer, "SPACE_SOURCE", source):
                with self.assertRaisesRegex(RuntimeError, "synthetic_ood_04"):
                    deployer.preflight()


if __name__ == "__main__":
    unittest.main()
