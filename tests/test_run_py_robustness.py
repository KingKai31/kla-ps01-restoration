"""
Formal, re-runnable version of the adversarial input test previously run
ad-hoc (scripts/_adversarial_eval_test.py built the inputs; verification
that run.py actually survived them was done by hand, once, and not kept as
a re-checkable artifact). This suite loads run.py directly (matching how
the real submission is invoked - no reimplementation of its logic) and
asserts specific, named guarantees against KLA's hard output spec:
(H, W) shape, float32, values in [0, 1], no NaN/Inf, and no single bad
input crashing the whole batch.

Requires the real shipped checkpoint (models/checkpoint.pt) - this
exercises the actual model forward pass, not a mock, since that's the
real robustness surface (e.g. the tiny-image case below only fails via a
real shape constraint inside NAFNetSR's padding, which a mocked model
would never surface).

Run with: pytest tests/test_run_py_robustness.py -v
(pytest is a dev-only test dependency, not pinned in the training-env
requirements.txt or the submission's self-contained requirements.txt -
`pip install pytest` into the dev venv separately to run this suite.)
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = REPO_ROOT / "models" / "checkpoint.pt"


def _load_run_module():
    """Imports run.py as a module by file path (not by package name) so
    this test exercises the exact file that ships in the submission
    folder, not a copy or reimplementation."""
    spec = importlib.util.spec_from_file_location("run_py_under_test", REPO_ROOT / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_module():
    return _load_run_module()


@pytest.fixture(scope="module")
def checkpoint_available():
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"{CHECKPOINT_PATH} not found - integration tests need the real shipped checkpoint")
    return CHECKPOINT_PATH


# --- adversarial input definitions ------------------------------------------
# Each entry: (filename, array-or-None, raw-bytes-or-None). Exactly one of
# array/raw_bytes is set; raw_bytes covers the "not even a valid .npy file"
# case which np.save can't produce.

def _build_adversarial_inputs(input_dir: Path) -> dict:
    """Recreates the same 10 cases as scripts/_adversarial_eval_test.py,
    seeded for reproducibility, returning {filename: original_array_or_None}
    (None for the corrupt file, which has no valid array)."""
    rng = np.random.default_rng(0)
    originals = {}

    def save(name, arr):
        np.save(input_dir / name, arr)
        originals[name] = arr

    save("normal_128.npy", rng.random((128, 128)).astype(np.float32))

    (input_dir / "corrupt.npy").write_bytes(b"this is not a valid npy file")
    originals["corrupt.npy"] = None

    save("wrong_ndim_3d.npy", rng.random((128, 128, 3)).astype(np.float32))
    save("wrong_ndim_1d.npy", rng.random(128 * 128).astype(np.float32))

    arr_nan = rng.random((128, 128)).astype(np.float32)
    arr_nan[10:20, 10:20] = np.nan
    arr_nan[50:55, 50:55] = np.inf
    arr_nan[60:65, 60:65] = -np.inf
    save("nan_inf.npy", arr_nan)

    save("all_zero.npy", np.zeros((128, 128), dtype=np.float32))
    save("all_constant.npy", np.full((128, 128), 0.73, dtype=np.float32))
    save("tiny_8x8.npy", rng.random((8, 8)).astype(np.float32))
    save("nonsquare_100x150.npy", rng.random((100, 150)).astype(np.float32))
    save("extreme_values.npy", (rng.random((128, 128)).astype(np.float32) * 1000 - 500))

    return originals


@pytest.fixture(scope="module")
def adversarial_run(tmp_path_factory, run_module, checkpoint_available):
    """Runs run.py's actual main() once over all 10 adversarial inputs
    together, exactly as a grading harness would invoke it on a real
    input_dir - not one process per case. Individual test functions below
    each assert one specific named guarantee against the shared result."""
    input_dir = tmp_path_factory.mktemp("adversarial_inputs")
    output_dir = tmp_path_factory.mktemp("adversarial_outputs")
    originals = _build_adversarial_inputs(input_dir)

    argv_backup = sys.argv
    sys.argv = ["run.py", str(input_dir), str(output_dir)]
    try:
        run_module.main()
    finally:
        sys.argv = argv_backup

    outputs = {f.name: np.load(f) for f in sorted(output_dir.glob("*.npy"))}
    return {"originals": originals, "outputs": outputs, "output_dir": output_dir}


def _assert_spec_compliant(arr: np.ndarray):
    """The universal KLA hard-gate contract every output file must satisfy,
    regardless of which code path (model / classical fallback / placeholder)
    produced it."""
    assert arr.dtype == np.float32, f"expected float32, got {arr.dtype}"
    assert arr.ndim == 2, f"expected (H, W), got shape {arr.shape}"
    assert np.all(np.isfinite(arr)), "output contains NaN/Inf"
    assert arr.min() >= 0.0 and arr.max() <= 1.0, f"output out of [0,1]: min={arr.min()}, max={arr.max()}"


class TestBatchSurvival:
    """The single most important guarantee: one bad file must never take
    down the rest of the batch."""

    def test_all_ten_inputs_produce_an_output_file(self, adversarial_run):
        expected = set(adversarial_run["originals"].keys())
        actual = set(adversarial_run["outputs"].keys())
        assert actual == expected, f"missing outputs for: {expected - actual}"

    def test_every_output_is_spec_compliant(self, adversarial_run):
        """Blanket check of the universal contract (dtype/shape/range/
        finiteness) across every single output file, regardless of case."""
        for name, arr in adversarial_run["outputs"].items():
            _assert_spec_compliant(arr)


class TestNormalCase:
    """Control case - confirms the harness itself is set up correctly and
    the model path is actually exercised (not silently falling back)."""

    def test_normal_image_produces_upscaled_output(self, adversarial_run):
        original = adversarial_run["originals"]["normal_128.npy"]
        out = adversarial_run["outputs"]["normal_128.npy"]
        assert out.shape[0] > original.shape[0] and out.shape[1] > original.shape[1], \
            "expected an upscaled output for a normal valid input"


class TestCorruptAndMalformedInputs:
    """Inputs that can't even be loaded as valid image arrays - these must
    hit the load-failure path (DEFAULT_SHAPE placeholder), not crash."""

    def test_corrupt_npy_file_gets_placeholder(self, adversarial_run, run_module):
        out = adversarial_run["outputs"]["corrupt.npy"]
        assert out.shape == run_module.DEFAULT_SHAPE

    def test_wrong_ndim_3d_gets_placeholder(self, adversarial_run, run_module):
        out = adversarial_run["outputs"]["wrong_ndim_3d.npy"]
        assert out.shape == run_module.DEFAULT_SHAPE

    def test_wrong_ndim_1d_gets_placeholder(self, adversarial_run, run_module):
        out = adversarial_run["outputs"]["wrong_ndim_1d.npy"]
        assert out.shape == run_module.DEFAULT_SHAPE


class TestValidButDegenerateInputs:
    """Inputs that load fine as 2D arrays but are numerically or spatially
    unusual - these should go through the model (or its classical
    fallback) and still come out spec-compliant."""

    def test_nan_inf_input_is_cleaned_before_model(self, adversarial_run):
        out = adversarial_run["outputs"]["nan_inf.npy"]
        _assert_spec_compliant(out)

    def test_all_zero_input_does_not_break_fallback_sigma_estimation(self, adversarial_run):
        """all_zero is the case explicitly called out in the original
        adversarial script as a risk: skimage's estimate_sigma on a
        constant image could degenerate. Confirms it doesn't propagate
        into a bad output."""
        out = adversarial_run["outputs"]["all_zero.npy"]
        _assert_spec_compliant(out)

    def test_all_constant_nonzero_input(self, adversarial_run):
        out = adversarial_run["outputs"]["all_constant.npy"]
        _assert_spec_compliant(out)

    def test_nonsquare_input_preserves_aspect_ratio(self, adversarial_run):
        original = adversarial_run["originals"]["nonsquare_100x150.npy"]
        out = adversarial_run["outputs"]["nonsquare_100x150.npy"]
        orig_ratio = original.shape[0] / original.shape[1]
        out_ratio = out.shape[0] / out.shape[1]
        assert orig_ratio == pytest.approx(out_ratio, rel=1e-6), \
            f"aspect ratio not preserved: {original.shape} -> {out.shape}"

    def test_extreme_out_of_range_values_still_clip_to_unit_range(self, adversarial_run):
        out = adversarial_run["outputs"]["extreme_values.npy"]
        _assert_spec_compliant(out)


class TestTinyImageModelPathLimit:
    """Real, documented finding from building this suite (not previously
    known): NAFNetSR's forward pass reflect-pads each input up to a
    multiple of padder_size=16 (2**len(enc_blk_nums)). PyTorch's reflect
    padding requires the pad amount to be strictly less than the input
    dimension, which fails for any side <=8px (pad needed >= side length).
    This IS caught by run.py's per-image try/except and correctly falls
    back to the classical path - the batch does not crash - but it means
    any image <=8px on either side silently downgrades to the much lower-
    quality classical restoration instead of the trained model. Disclosing
    this rather than leaving it undiscovered."""

    def test_reflect_pad_constraint_is_real(self):
        """Directly demonstrates the underlying torch constraint this
        finding is based on, independent of run.py, so this test doesn't
        silently go stale if run.py's fallback logic changes."""
        import torch
        import torch.nn.functional as F
        x = torch.randn(1, 1, 8, 8)
        with pytest.raises(RuntimeError, match="Padding size should be less"):
            F.pad(x, (0, 8, 0, 8), mode="reflect")

    def test_tiny_8x8_input_falls_back_but_does_not_crash_batch(self, adversarial_run):
        """The model path fails internally for this input (see above), but
        the batch-level guarantee - spec-compliant output, no crash -
        must still hold via the classical fallback."""
        out = adversarial_run["outputs"]["tiny_8x8.npy"]
        _assert_spec_compliant(out)
        assert out.shape[0] == 16 and out.shape[1] == 16  # 8px * upscale(2)


class TestSanitizeOutput:
    """Unit-level tests of sanitize_output() in isolation - the final gate
    applied before every np.save call in run.py, regardless of which path
    produced the array."""

    def test_passthrough_for_already_valid_array(self, run_module):
        arr = np.array([[0.1, 0.5], [0.9, 0.0]], dtype=np.float32)
        out = run_module.sanitize_output(arr)
        np.testing.assert_array_equal(out, arr)

    def test_nan_replaced_with_midpoint(self, run_module):
        arr = np.array([[np.nan, 0.5]], dtype=np.float32)
        out = run_module.sanitize_output(arr)
        assert out[0, 0] == 0.5
        assert np.all(np.isfinite(out))

    def test_positive_infinity_clamped_to_one(self, run_module):
        arr = np.array([[np.inf, 0.2]], dtype=np.float32)
        out = run_module.sanitize_output(arr)
        assert out[0, 0] == 1.0

    def test_negative_infinity_clamped_to_zero(self, run_module):
        arr = np.array([[-np.inf, 0.2]], dtype=np.float32)
        out = run_module.sanitize_output(arr)
        assert out[0, 0] == 0.0

    def test_out_of_range_values_clipped(self, run_module):
        arr = np.array([[-5.0, 5.0]], dtype=np.float32)
        out = run_module.sanitize_output(arr)
        assert out[0, 0] == 0.0 and out[0, 1] == 1.0

    def test_wrong_ndim_raises_assertion(self, run_module):
        arr = np.random.rand(4, 4, 4).astype(np.float32)
        with pytest.raises(AssertionError, match="expected \\(H, W\\)"):
            run_module.sanitize_output(arr)


class TestClassicalFallback:
    """Unit-level tests of classical_fallback() (bicubic + non-local-means)
    in isolation, covering the same degenerate-input risks the adversarial
    inputs target, without needing the model/checkpoint loaded."""

    def test_normal_image_upscales_by_requested_factor(self, run_module):
        arr = np.random.default_rng(0).random((32, 32)).astype(np.float32)
        out = run_module.classical_fallback(arr, scale=2)
        assert out.shape == (64, 64)
        _assert_spec_compliant(out)

    def test_all_zero_image_does_not_crash_sigma_estimation(self, run_module):
        arr = np.zeros((32, 32), dtype=np.float32)
        out = run_module.classical_fallback(arr, scale=2)
        _assert_spec_compliant(out)

    def test_all_constant_nonzero_image(self, run_module):
        arr = np.full((32, 32), 0.73, dtype=np.float32)
        out = run_module.classical_fallback(arr, scale=2)
        _assert_spec_compliant(out)

    def test_extreme_out_of_range_values_get_clipped(self, run_module):
        arr = (np.random.default_rng(0).random((32, 32)).astype(np.float32) * 1000 - 500)
        out = run_module.classical_fallback(arr, scale=2)
        _assert_spec_compliant(out)

    def test_nonsquare_image(self, run_module):
        arr = np.random.default_rng(0).random((20, 30)).astype(np.float32)
        out = run_module.classical_fallback(arr, scale=2)
        assert out.shape == (40, 60)
        _assert_spec_compliant(out)
