from pathlib import Path

from simulation_with_orientation import main as run_simulation


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    output_dir = PACKAGE_ROOT / "outputs" / "simulation_example"
    cif_dir = Path(__file__).resolve().parent / "cif"
    run_simulation(
        dataset_folder=str(output_dir),
        cif_folder=str(cif_dir),
        cif_name="Ag.cif",
        cif_index=0,
        cif_count=1,
        flag_check=False,
        simulation_count=1,
    )
