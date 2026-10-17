import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    spec = importlib.util.spec_from_file_location("build_notebook", ROOT / "scripts" / "build_notebook.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_notebook_is_up_to_date():
    builder = load_builder()
    on_disk = (ROOT / "notebooks" / "colab_runner.ipynb").read_text(encoding="utf-8")
    assert on_disk == builder.render(), "run: python scripts/build_notebook.py"


def test_notebook_cells_reproduce_repository_files():
    """Each %%writefile cell must write exactly the repository file it names."""
    nb = json.loads((ROOT / "notebooks" / "colab_runner.ipynb").read_text(encoding="utf-8"))
    written = {}
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and src.startswith("%%writefile "):
            first, body = src.split("\n", 1)
            path = first.split(" ", 1)[1]
            assert body.splitlines()[0] == f"# Repository file: {path}"
            written[path] = body
    builder = load_builder()
    assert set(written) == {p for p, _ in builder.SOURCE_FILES + builder.EVAL_FILES}
    for path, body in written.items():
        assert body.rstrip("\n") == (ROOT / path).read_text(encoding="utf-8").rstrip("\n"), path
