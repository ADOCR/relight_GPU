"""Validate merged CLI exports with synthetic input; no timing claims.

Usage: python tests/integration/verify_export_merge.py CLI FIXTURE OUTPUT
Dependencies: benchmarks/requirements.txt. Generate FIXTURE with generate_fixture.py.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from run_export_benchmark import compare, load_output


def validate(binary, fixture, output):
    metadata = json.loads((fixture / "fixture.json").read_text())
    assert metadata["synthetic"] and metadata["images"] >= 27
    assert metadata["width"] >= 46 and metadata["height"] >= 18
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(RELIGHT_PROFILE="1", OMP_NUM_THREADS="2")
    results = []
    cases = [
        ("ptm", 18, ["-H", "-k", "37x13+5+3"], (37, 13), True),
        ("hsh", 27, [], (metadata["width"], metadata["height"]), True),
        ("rbf", 9, ["-k", "37x13+5+3"], (37, 13), False),
        ("bilinear", 9, ["-k", "39x13+5+3"], (39, 13), False),
    ]
    for basis, planes, flags, dimensions, accelerated in cases:
        profiles = []
        for backend, workers in (("cpu", 1), ("cuda", 2)):
            folder = output / f"{basis}_{backend}"
            command = [str(binary), "-b", basis, "-p", str(planes), "-s", "16",
                       "-w", str(workers), "-g", backend, *flags, str(fixture), str(folder)]
            process = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, timeout=180)
            (output / f"{basis}_{backend}.log").write_text(process.stdout, encoding="utf-8")
            assert process.returncode == 0, process.stdout[-4000:]
            lines = [line[10:] for line in process.stdout.splitlines() if line.startswith("[profile] ")]
            assert lines, "Missing backend profile"
            profile = json.loads(lines[-1])
            profiles.append(profile)
            info, _, _, _ = load_output(folder)
            assert (info["width"], info["height"]) == dimensions
            assert info["nplanes"] == planes
            assert len(info["lights"]) == 3 * metadata["images"], "Input light list was reduced"
            assert (profile["gpu_projection_calls"] > 0) == (backend == "cuda" and accelerated)
        difference = compare(output / f"{basis}_cpu", output / f"{basis}_cuda")
        if not accelerated:
            # Both runs use CPU; different row-worker counts must not alter RBF results.
            assert difference["coefficient_abs_max"] == 0, difference
        result = {"basis": basis, "planes": planes, "dimensions": list(dimensions),
                  "expected_cuda": accelerated, "comparison": difference,
                  "gpu_projection_calls": profiles[1]["gpu_projection_calls"]}
        results.append(result)
        print(json.dumps(result), flush=True)
    result = {"synthetic": True, "input_images": metadata["images"],
              "checks": results, "accuracy_validation_only": True}
    (output / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    validate(args.binary.resolve(), args.fixture.resolve(), args.output.resolve())
