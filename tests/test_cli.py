import subprocess


def test_vexrag_invokation():
    invokation_result: subprocess.CompletedProcess = subprocess.run(
        "vexrag",
        shell=True,
        capture_output=True,
    )
    assert invokation_result.returncode == 0

