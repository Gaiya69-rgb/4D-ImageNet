# Metadata Template

This file is intended as a compact metadata directory for the delivered 4D-STEM dataset.


## 1. Instrument Information

- Microscope model: Talos F200X G2
- Detector model: Dectris Arina
- Accelerating voltage: 200KV
- Camera length calibration: Null
- Convergence semi-angle calibration: Null
- Flatfield correction: Flatfield data are provided and the calibration is carried out in "transform_format.py"
- sample preparation: well-dispersed nanoparticles


## 2. Filename Parameter Description

Merged files are named using the acquisition parameter string from the original source file.

Example:

```text
Ag_FOV190nm_CL160mm_260_260_1.5ms_2.1mrad_650kx_000_master.pkl
```

Parameter description:

- `Ag`
  Material name.

- `FOV190nm`
  Field of view in real space.

- `CL160mm`
  Camera length.

- `260_260`
  Scan grid size in real space, usually written as `scan_x_scan_y`.

- `1.5ms`
  Dwell time or exposure time per scan position.

- `2.1mrad`
  Convergence semi-angle or acquisition-angle-related setting.

- `650kx`
  Magnification label from the acquisition filename.

- `000`
  Acquisition index / repeat index.

- `sp7`
We used spot size 7 if mentioned and the default value is 9


