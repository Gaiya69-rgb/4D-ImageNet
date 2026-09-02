# Data-preparation utilities

The Python files in this directory are byte-identical copies of the maintained
project implementations. They provide the non-model stages described in the data
descriptor:

- `transform_format.py`: ARINA H5 processing and Merlin MIB-to-H5 conversion,
  with a Tkinter GUI and a programmatic entry point.
- `extract_pkl.py`: beam-centre estimation, diffraction-pattern scoring, and
  top-5,000 experimental pattern extraction.
- `merge_pkl.py`: rank-ordered merging of individual pattern records into one
  acquisition-level PKL without changing detector intensities.
- `simulation_with_orientation.py`: abTEM multislice simulation with Euler-angle
  and approximate hkl metadata.
- `audit_4d_imagenet.py`: structural and count auditing of the released records.
- `visualize_pkl.py` and `visualize_merged_pkl.py`: inspection utilities.

The `cif/` folder contains the 14 CIF inputs present in the project. Select
**Transform Format - GUI** in VS Code for format conversion. The package-relative
`run_simulation_example.py` wrapper requests one Ag simulation from the unchanged
simulation implementation; it requires the optional data-pipeline environment
and a compatible GPU setup.

Raw detector files are not embedded in this baseline package. Format conversion
and full experimental curation therefore require user-supplied ARINA or Merlin
data. The files under `sample_data/` are compact downstream smoke-test inputs.
