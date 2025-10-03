import os
from bench import plot_results

def test_plot_results(tmp_path):
    # Run plotting
    json_path = os.path.join("glitchlab", "gui", "bench", "artifacts", "ab.json")
    out_dir = tmp_path / "plots"
    plot_results(json_path=json_path, out_dir=out_dir)

    # Assert files exist
    assert (out_dir / "accuracy.png").exists()
    assert (out_dir / "timings.png").exists()
    assert (out_dir / "align_vs_ast.png").exists()
