from pathlib import Path


def test_migration_control_files_exist():
    root = Path(__file__).resolve().parents[1]
    expected = [
        "AGENTS.md",
        "PLANS.md",
        "CMakeLists.txt",
        "README_mfem.md",
        "bindings/pybind_module.cpp",
        "apps/mt_forward_cli.cpp",
        "legacy_cuda/forward_module.cpp",
        "legacy_cuda/setup.py",
    ]

    missing = [path for path in expected if not (root / path).exists()]
    assert missing == []
