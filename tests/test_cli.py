import subprocess
import sys


def test_vexrag_invokation():
    invokation_result: subprocess.CompletedProcess = subprocess.run(
        [
            sys.executable,
            "-m",
            "vexrag",
        ],
        capture_output=True,
        text=True,
    )
    assert invokation_result.returncode == 0
    assert "ready" in invokation_result.stdout.lower()
