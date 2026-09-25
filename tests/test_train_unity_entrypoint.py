from pathlib import Path
import subprocess
import sys
import unittest


class TrainUnityEntrypointTest(unittest.TestCase):
    """Break caught: path rewriting works in package tests but not script execution."""

    def test_hydra_config_dry_run_imports_successfully(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [
                sys.executable,
                "dual2pose/train_unity.py",
                "--cfg",
                "job",
            ],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("seed: 42", result.stdout)
        self.assertIn("test_ckpt_path: best", result.stdout)


if __name__ == "__main__":
    unittest.main()
