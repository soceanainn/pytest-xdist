from __future__ import annotations

import platform
import sys
from types import SimpleNamespace

import pytest

from xdist.remote import apply_worker_memory_limit


GIB = 1024**3
MEMORY_LIMIT_SUPPORTED = platform.system() == "Linux" or (
    platform.system() == "Darwin" and int(platform.release().split(".")[0]) >= 21
)
# Apple silicon reserves hundreds of GiB of virtual address space at startup.
TEST_LIMIT_GIB = 512 if platform.system() == "Darwin" else 64


@pytest.mark.parametrize(
    ("ini_value", "args", "expected"),
    [
        (None, [], 0),
        ("64", [], 64),
        (None, ["--worker-max-virtual-memory-gib=32"], 32),
        ("64", ["--worker-max-virtual-memory-gib=32"], 32),
        ("64", ["--worker-max-virtual-memory-gib=0"], 0),
    ],
)
def test_memory_limit_configuration(
    pytester: pytest.Pytester,
    ini_value: str | None,
    args: list[str],
    expected: int,
) -> None:
    if ini_value is not None:
        pytester.makeini(f"[pytest]\nworker_max_virtual_memory_gib = {ini_value}")
    config = pytester.parseconfigure(*args)
    assert config.getoption("worker_max_virtual_memory_gib") == expected


@pytest.mark.parametrize("value", ["", "-1", "1.5", "64G", "nan", str(sys.maxsize)])
@pytest.mark.parametrize("source", ["ini", "cli"])
def test_invalid_memory_limit(
    pytester: pytest.Pytester, value: str, source: str
) -> None:
    args = []
    if source == "ini":
        pytester.makeini(f"[pytest]\nworker_max_virtual_memory_gib = {value}")
    else:
        args.append(f"--worker-max-virtual-memory-gib={value}")
    with pytest.raises(pytest.UsageError, match="worker_max_virtual_memory_gib"):
        pytester.parseconfigure(*args)


@pytest.mark.parametrize(
    ("system", "release"), [("Linux", "6.0"), ("Darwin", "21.0"), ("Darwin", "25.0")]
)
@pytest.mark.parametrize(
    ("soft", "hard", "expected"),
    [
        (-1, -1, (2 * GIB, 2 * GIB)),
        (GIB, -1, (GIB, 2 * GIB)),
        (3 * GIB, 4 * GIB, (2 * GIB, 2 * GIB)),
        (GIB, GIB, None),
        (GIB, 2 * GIB, None),
        (0, -1, (0, 2 * GIB)),
    ],
)
def test_apply_memory_limit(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    release: str,
    soft: int,
    hard: int,
    expected: tuple[int, int] | None,
) -> None:
    calls: list[tuple[int, tuple[int, int]]] = []
    resource = SimpleNamespace(
        RLIMIT_AS=9,
        RLIM_INFINITY=-1,
        getrlimit=lambda kind: (soft, hard),
        setrlimit=lambda kind, limits: calls.append((kind, limits)),
    )
    monkeypatch.setitem(sys.modules, "resource", resource)
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "release", lambda: release)

    apply_worker_memory_limit(2)

    assert calls == ([] if expected is None else [(resource.RLIMIT_AS, expected)])


@pytest.mark.parametrize(
    ("system", "release", "max_gib"),
    [
        ("Linux", "6.0", 0),
        ("Darwin", "21.0", 0),
        ("Darwin", "20.0", 64),
        ("Windows", "10", 64),
    ],
)
def test_memory_limit_disabled_or_unsupported(
    monkeypatch: pytest.MonkeyPatch, system: str, release: str, max_gib: int
) -> None:
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "release", lambda: release)
    monkeypatch.setitem(sys.modules, "resource", None)

    apply_worker_memory_limit(max_gib)


def test_macos_memory_limit_error_explains_virtual_reservations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def setrlimit(kind: int, limits: tuple[int, int]) -> None:
        raise ValueError("current limit exceeds maximum limit")

    monkeypatch.setitem(
        sys.modules,
        "resource",
        SimpleNamespace(
            RLIMIT_AS=5,
            RLIM_INFINITY=-1,
            getrlimit=lambda kind: (-1, -1),
            setrlimit=setrlimit,
        ),
    )
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "release", lambda: "21.0")

    with pytest.raises(
        ValueError, match=r"macOS rejects limits below.*virtual address"
    ):
        apply_worker_memory_limit(64)


@pytest.mark.parametrize(
    ("ini_value", "args", "expected"),
    [
        (str(TEST_LIMIT_GIB), [], TEST_LIMIT_GIB),
        ("0", [f"--worker-max-virtual-memory-gib={TEST_LIMIT_GIB}"], TEST_LIMIT_GIB),
        (str(TEST_LIMIT_GIB), ["--worker-max-virtual-memory-gib=0"], 0),
    ],
)
def test_memory_limit_in_workers(
    pytester: pytest.Pytester, ini_value: str, args: list[str], expected: int
) -> None:
    # Compare against inherited limits, including on hosts with stricter limits.
    if MEMORY_LIMIT_SUPPORTED:
        import resource

        original = resource.getrlimit(resource.RLIMIT_AS)
    else:
        original = None
    pytester.makeini(f"[pytest]\nworker_max_virtual_memory_gib = {ini_value}")
    pytester.makeconftest(
        f"""
        import os

        # The cap must already apply while importing conftest, before collection.
        if {MEMORY_LIMIT_SUPPORTED!r}:
            import resource

            original = {original!r}
            expected = original
            if os.environ.get("PYTEST_XDIST_WORKER") and {expected}:
                cap = {expected} * 1024**3
                expected = tuple(
                    cap if limit == resource.RLIM_INFINITY else min(limit, cap)
                    for limit in original
                )
            assert resource.getrlimit(resource.RLIMIT_AS) == expected
        """
    )
    pytester.makepyfile(
        f"""
        def test_limit(request):
            assert request.config.workerinput["worker_max_virtual_memory_gib"] == {expected}
        """
    )

    result = pytester.runpytest_subprocess("-n2", *args)

    result.assert_outcomes(passed=1)


@pytest.mark.skipif(not MEMORY_LIMIT_SUPPORTED, reason="Requires Linux or macOS 12+")
def test_runaway_allocation_fails_in_worker(pytester: pytest.Pytester) -> None:
    limit = 512 if platform.system() == "Darwin" else 4
    pytester.makepyfile(
        f"""
        import mmap

        def test_runaway():
            # Reserve address space without committing gigabytes of physical RAM.
            with mmap.mmap(-1, {limit} * 1024**3):
                pass

        def test_healthy():
            assert bytearray(1024)
        """
    )

    result = pytester.runpytest_subprocess(
        "-n1", f"--worker-max-virtual-memory-gib={limit}"
    )

    result.assert_outcomes(failed=1, passed=1)
    result.stdout.fnmatch_lines(["*test_runaway*", "*OSError*Cannot allocate memory*"])


def test_replacement_worker_keeps_memory_limit(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        f"""
        import os

        def test_crash():
            os._exit(1)

        def test_replacement(worker_id, request):
            assert worker_id == "gw1"
            assert request.config.workerinput["worker_max_virtual_memory_gib"] == {TEST_LIMIT_GIB}
            if {MEMORY_LIMIT_SUPPORTED!r}:
                import resource

                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                assert 0 <= soft <= hard <= {TEST_LIMIT_GIB} * 1024**3
        """
    )

    result = pytester.runpytest_subprocess(
        "-n1", f"--worker-max-virtual-memory-gib={TEST_LIMIT_GIB}"
    )

    result.assert_outcomes(failed=1, passed=1)


def test_memory_limit_does_not_apply_without_workers(pytester: pytest.Pytester) -> None:
    if MEMORY_LIMIT_SUPPORTED:
        import resource

        original = resource.getrlimit(resource.RLIMIT_AS)
    else:
        original = None
    pytester.makepyfile(
        f"""
        def test_no_limit(worker_id):
            assert worker_id == "master"
            if {MEMORY_LIMIT_SUPPORTED!r}:
                import resource

                assert resource.getrlimit(resource.RLIMIT_AS) == {original!r}
        """
    )

    result = pytester.runpytest_subprocess("-n0", "--worker-max-virtual-memory-gib=64")

    result.assert_outcomes(passed=1)


@pytest.mark.skipif(
    platform.system() != "Darwin" or not MEMORY_LIMIT_SUPPORTED,
    reason="Requires macOS 12+",
)
def test_macos_limit_below_startup_usage(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("def test_healthy(): pass")

    result = pytester.runpytest_subprocess(
        "-n1", "--worker-max-virtual-memory-gib=1", "--max-worker-restart=0"
    )

    assert result.ret != 0
    assert "macOS rejects limits below" in result.stdout.str()
    assert "Choose a higher limit or use 0 to disable it" in result.stdout.str()
