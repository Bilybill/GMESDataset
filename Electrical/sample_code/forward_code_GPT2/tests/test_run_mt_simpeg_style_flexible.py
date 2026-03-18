from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.run_mt_simpeg_style_flexible import run


def test_run_mt_simpeg_style_flexible(tmp_path):
    data, outdir = run(plotIt=False, save_outputs=True, outdir=tmp_path / 'demo')
    assert data.dobs.numel() > 0
    assert (outdir / 'survey_layout.png').exists()
    assert (outdir / 'central_station_rho_phase.png').exists()
    assert (outdir / 'predicted_data.npy').exists()
    assert (outdir / 'receiver_locations.npy').exists()
    assert (outdir / 'central_station_mt_curves.npz').exists()
    assert (outdir / 'metadata.json').exists()
