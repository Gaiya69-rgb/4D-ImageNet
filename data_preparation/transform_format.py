#Libraries import

import numpy as np

import tifffile
import os
import math

import tkinter as tk
from tkinter import filedialog, messagebox
import re
import pathlib
import argparse

try:
    import h5py
except ImportError:  # pragma: no cover - h5py expected in py4DSTEM env
    h5py = None

try:
    import py4DSTEM
except ImportError:  # pragma: no cover - optional for Merlin MIB conversion
    py4DSTEM = None


class PathInputApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Path_input")
        self.root.geometry("500x200")

        # 标签和输入框
        tk.Label(root, text="src_path:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.src_entry = tk.Entry(root, width=40)
        self.src_entry.grid(row=0, column=1)
        tk.Button(root, text="Browser", command=self.browse_src).grid(row=0, column=2)

        tk.Label(root, text="dest_path:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
        self.dest_entry = tk.Entry(root, width=40)
        self.dest_entry.grid(row=1, column=1)
        tk.Button(root, text="Browser", command=self.browse_dest).grid(row=1, column=2)

        tk.Label(root, text="flatfield_path:").grid(row=2, column=0, padx=10, pady=10, sticky="e")
        self.flat_entry = tk.Entry(root, width=40)
        self.flat_entry.grid(row=2, column=1)
        tk.Button(root, text="Browser", command=self.browse_flat).grid(row=2, column=2)

        # 提交按钮
        tk.Button(root, text="Submit", command=self.submit).grid(row=3, column=1, pady=20)

    def browse_src(self):
        # Allow selecting either a folder, a single ARINA master.h5 file, or a Merlin mib file
        path = filedialog.askopenfilename(
            title="Choose src folder, master.h5, or mib",
            filetypes=[("4D-STEM files", "*.h5 *.mib"), ("HDF5/NeXus", "*.h5"), ("Merlin MIB", "*.mib"), ("All files", "*.*")])
        if not path:
            path = filedialog.askdirectory(title="Choose src folder")
        if path:
            self.src_entry.delete(0, tk.END)
            self.src_entry.insert(0, path)

    def browse_dest(self):
        path = filedialog.askdirectory(title="Choose dest_path")
        if path:
            self.dest_entry.delete(0, tk.END)
            self.dest_entry.insert(0, path)

    def browse_flat(self):
        """Allow choosing TIFF or master HDF5 flatfield files."""
        path = filedialog.askopenfilename(
            title="Choose flatfield_path",
            filetypes=[
                ("ARINA master files", "*_master.h5"),
                ("HDF5 Files", "*.h5 *.hdf5"),
                ("TIFF Files", "*.tif *.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.flat_entry.delete(0, tk.END)
            self.flat_entry.insert(0, path)

    def submit(self):
        src_path = self.src_entry.get().strip()
        dest_path = self.dest_entry.get().strip()
        flat_path = self.flat_entry.get().strip()

        if not src_path or not dest_path:
            messagebox.showwarning("Warning", "Please fill in src_path and dest_path!")
        else:
            copy_master_h5_files(src_path, dest_path, flat_path)


def defect_points_correction(data,flatfield):
    # define faltfiled == 0 is defect points
    defect_points = np.argwhere(flatfield == 0)

    # 8-neighbour
    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),          (0, 1),
                 (1, -1), (1, 0),  (1, 1)]

    for x, y in defect_points:
        vals = []
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < data.shape[0] and 0 <= ny < data.shape[1]:
                if flatfield[nx, ny] != 0:
                    vals.append(data[nx, ny])
        if vals:  
            data[x, y] = np.mean(vals)
        else:
            raise ValueError("The defect points are out of spec!")
    
    return data


def extract_number(token, label):
    match = re.search(r'(\d+)', token)
    if not match:
        raise ValueError(f"Cannot parse {label} from '{token}'")
    return int(match.group(1))


def _first_2d_dataset(h5_file):
    """Return all candidate 2D datasets found in an opened h5 file."""
    for _, obj in h5_file.items():
        if hasattr(obj, "shape") and len(obj.shape) == 2:
            yield obj
        elif hasattr(obj, "shape") and len(obj.shape) == 3 and obj.shape[0] == 1:
            yield obj[0]  # squeeze leading singleton
        if hasattr(obj, "items"):
            yield from _first_2d_dataset(obj)


def load_flatfield(path):
    """Load flatfield from .tif/.tiff or .h5/.hdf5."""
    ext = pathlib.Path(path).suffix.lower()
    if ext in [".tif", ".tiff"]:
        return tifffile.imread(path).astype("float32")
    if ext in [".h5", ".hdf5"]:
        if h5py is None:
            raise ImportError("h5py is required to read HDF5 flatfield files")
        with h5py.File(path, "r") as f:
            candidates = []
            for ds in _first_2d_dataset(f):
                # Normalize to ndarray so we can examine shape safely
                arr = np.array(ds)
                if arr.ndim != 2:
                    continue
                h, w = arr.shape
                if min(h, w) < 16:
                    continue  # likely a coordinate table, not an image
                area = h * w
                squareness = min(h, w) / max(h, w)
                candidates.append((area, squareness, arr))

            if not candidates:
                raise ValueError("Cannot find usable 2D dataset in flatfield HDF5 file")

            # Prefer largest area; tie-break by squareness to avoid thin tables
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return candidates[0][2].astype("float32")
    raise ValueError(f"Unsupported flatfield format: {ext}")


def infer_scan_shape(master_path, guessed_width=None, guessed_height=None):
    """Infer scan width/height from number of images stored in the ARINA master file."""
    if h5py is None:
        raise ImportError("h5py is required to infer scan shape from master files")

    with h5py.File(master_path, "r") as f:
        data_group = f["entry"]["data"]
        nimages = sum(data_group[name].shape[0] for name in data_group)

    guessed_width = int(guessed_width) if guessed_width else None
    guessed_height = int(guessed_height) if guessed_height else None

    def try_width(width):
        if width and width > 0 and nimages % width == 0:
            return int(width), int(nimages // width)
        return None

    def try_height(height):
        if height and height > 0 and nimages % height == 0:
            return int(nimages // height), int(height)
        return None

    for attempt in (try_width(guessed_width), try_height(guessed_height)):
        if attempt:
            return attempt

    # Fallback: search factors near sqrt(nimages) and choose the orientation
    # closest to the provided guesses.
    def closest_factor_pair():
        root = math.isqrt(nimages)
        for lower in range(root, 0, -1):
            if nimages % lower == 0:
                return int(lower), int(nimages // lower)
        return None, None

    low, high = closest_factor_pair()
    if not low or not high:
        raise ValueError(f"Cannot infer scan dimensions from master file with {nimages} frames")

    orientations = [(low, high)]
    if low != high:
        orientations.append((high, low))

    if guessed_width or guessed_height:
        def score(option):
            width_score = abs(option[0] - guessed_width) if guessed_width else 0
            height_score = abs(option[1] - guessed_height) if guessed_height else 0
            return width_score + height_score

        orientations.sort(key=score)
        return orientations[0]

    # Default to width >= height when no hints are available.
    return max(orientations, key=lambda opt: opt[0])

def parse_metadata_from_filename(srcfilename):
    """Parse scan size, FOV, camera length, and unit from an ARINA filename."""
    name = os.path.basename(srcfilename)
    tokens = name.split('_')

    fov = None
    camera_length = None
    scan_dims = []

    if 'weim' in name:
        unit = 'weim'
    elif 'nm' in name:
        unit = 'nm'
    else:
        raise ValueError("Source filename does not contain 'weim' or 'nm'")

    for token in tokens:
        if 'FOV' in token:
            fov = extract_number(token, 'FOV')
        elif token.startswith('CL'):
            camera_length = extract_number(token, 'camera length')
        elif re.fullmatch(r'\d+', token):
            scan_dims.append(int(token))

    if len(scan_dims) < 2:
        raise ValueError(f"Cannot parse scan dimensions from '{name}'")

    if fov is None:
        raise ValueError(f"Cannot parse FOV from '{name}'")
    if camera_length is None:
        raise ValueError(f"Cannot parse camera length from '{name}'")

    scan_width, scan_height = scan_dims[:2]
    return fov, camera_length, scan_width, scan_height, unit


def sanitize_filename_part(name):
    return re.sub(r"[^\w.\-]+", "_", name).strip("_")


def parse_merlin_metadata_from_filename(srcfilename):
    """Parse Merlin scan shape and useful acquisition metadata from a mib filename."""
    name = pathlib.Path(srcfilename).stem
    scan_match = re.search(r"(?P<x>\d+)\s*[xX]\s*(?P<y>\d+)", name)
    if not scan_match:
        raise ValueError(f"Cannot parse Merlin scan dimensions from '{srcfilename}'")

    step_match = re.search(r"ss(?P<step>\d+(?:\.\d+)?)\s*(?P<unit>nm|um|weim)", name, re.IGNORECASE)
    cl_match = re.search(r"CL(?P<cl>\d+(?:\.\d+)?)\s*mm", name, re.IGNORECASE)
    dwell_match = re.search(r"(?P<dwell>\d+(?:\.\d+)?)\s*ms", name, re.IGNORECASE)
    conv_match = re.search(r"(?P<conv>\d+(?:\.\d+)?)\s*mrad", name, re.IGNORECASE)

    scan_width = int(scan_match.group("x"))
    scan_height = int(scan_match.group("y"))
    step = float(step_match.group("step")) if step_match else None
    unit = step_match.group("unit") if step_match else "nm"
    camera_length = float(cl_match.group("cl")) if cl_match else None
    dwell_ms = float(dwell_match.group("dwell")) if dwell_match else None
    convergence_mrad = float(conv_match.group("conv")) if conv_match else None

    fov = None
    fov_unit = unit
    if step is not None:
        fov = step * scan_width

    return {
        "scan_width": scan_width,
        "scan_height": scan_height,
        "step": step,
        "step_unit": unit,
        "fov": fov,
        "fov_unit": fov_unit,
        "camera_length": camera_length,
        "dwell_ms": dwell_ms,
        "convergence_mrad": convergence_mrad,
    }


def parse_merlin_mib_header(header_bytes):
    """Parse the ASCII Merlin MIB frame header."""
    text = header_bytes.decode("latin1", errors="replace").split("\x00", 1)[0].strip()
    fields = [field.strip() for field in text.split(",")]
    if len(fields) < 7 or not fields[0].startswith("MQ"):
        raise ValueError("Input does not look like a Merlin MIB file")

    header_size = int(fields[2])
    detector_width = int(fields[4])
    detector_height = int(fields[5])
    pixel_type = fields[6].upper()
    dtype_map = {
        "U08": np.uint8,
        "U16": np.uint16,
        "U32": np.uint32,
        "I16": np.int16,
        "I32": np.int32,
    }
    if pixel_type not in dtype_map:
        raise ValueError(f"Unsupported Merlin MIB pixel type: {pixel_type}")

    exposure = float(fields[10]) if len(fields) > 10 and fields[10] else None
    return {
        "raw_header": text,
        "header_size": header_size,
        "detector_width": detector_width,
        "detector_height": detector_height,
        "pixel_type": pixel_type,
        "dtype": np.dtype(dtype_map[pixel_type]),
        "exposure_s": exposure,
    }


def read_merlin_mib_header(mib_path):
    with open(mib_path, "rb") as f:
        first = f.read(1024)
    preliminary = parse_merlin_mib_header(first[:384])
    header_size = preliminary["header_size"]
    if header_size != 384:
        with open(mib_path, "rb") as f:
            return parse_merlin_mib_header(f.read(header_size))
    return preliminary


def build_merlin_h5_name(srcfilename, metadata=None):
    """Keep the original Merlin filename stem and only change the extension to .h5."""
    return f"{pathlib.Path(srcfilename).stem}.h5"


def merlin_mib_to_h5(srcfilepath, srcfilename, destfilepath, destfilename=None, chunk_frames=None):
    """Convert a Merlin .mib file into a 4D HDF5 datacube."""
    if h5py is None:
        raise ImportError("h5py is required to write HDF5 files")

    metadata = parse_merlin_metadata_from_filename(srcfilename)
    mib_path = os.path.join(srcfilepath, srcfilename)
    header = read_merlin_mib_header(mib_path)

    scan_width = metadata["scan_width"]
    scan_height = metadata["scan_height"]
    detector_width = header["detector_width"]
    detector_height = header["detector_height"]
    dtype = header["dtype"]

    frame_pixels = detector_width * detector_height
    frame_bytes = header["header_size"] + frame_pixels * dtype.itemsize
    file_bytes = os.path.getsize(mib_path)
    if file_bytes % frame_bytes != 0:
        raise ValueError(
            f"MIB file size {file_bytes} is not divisible by frame size {frame_bytes}; "
            "header or pixel type may be unsupported."
        )

    nframes = file_bytes // frame_bytes
    expected_frames = scan_width * scan_height
    if nframes != expected_frames:
        raise ValueError(
            f"Filename scan shape expects {expected_frames} frames, but MIB contains {nframes} frames"
        )

    if destfilename is None:
        destfilename = build_merlin_h5_name(srcfilename, metadata)
    os.makedirs(destfilepath, exist_ok=True)
    out_path = os.path.join(destfilepath, destfilename)

    h5_dtype = dtype
    if chunk_frames is None:
        chunk_frames = scan_height
    with h5py.File(out_path, "w") as h5:
        datacube_root = h5.create_group("datacube_root")
        datacube_group = datacube_root.create_group("datacube")
        data = datacube_group.create_dataset(
            "data",
            shape=(scan_width, scan_height, detector_height, detector_width),
            dtype=h5_dtype,
            chunks=(1, scan_height, detector_height, detector_width),
        )

        entry = h5.create_group("entry")
        entry_data = entry.create_group("data")
        entry_data["data"] = data

        attrs = {
            "source_format": "Merlin MIB",
            "source_file": os.path.abspath(mib_path),
            "scan_width": scan_width,
            "scan_height": scan_height,
            "detector_width": detector_width,
            "detector_height": detector_height,
            "pixel_type": header["pixel_type"],
            "mib_header_size": header["header_size"],
            "real_space_unit": metadata.get("step_unit", "nm"),
        }
        for key, value in metadata.items():
            if value is not None:
                attrs[key] = value
        if header.get("exposure_s") is not None:
            attrs["exposure_s"] = header["exposure_s"]
        datacube_group.attrs.update(attrs)
        data.attrs.update(attrs)
        data.attrs["axes"] = "scan_x,scan_y,qy,qx"

        frames_per_read = max(1, int(chunk_frames))
        with open(mib_path, "rb") as f:
            for start in range(0, nframes, frames_per_read):
                count = min(frames_per_read, nframes - start)
                raw = f.read(count * frame_bytes)
                if len(raw) != count * frame_bytes:
                    raise EOFError(f"Unexpected end of file at frame {start}")

                records = np.frombuffer(raw, dtype=np.uint8).reshape(count, frame_bytes)
                pixel_bytes = records[:, header["header_size"]:]
                batch = pixel_bytes.copy().view(dtype).reshape(count, detector_height, detector_width)

                if start % scan_height == 0 and count % scan_height == 0:
                    row_start = start // scan_height
                    row_count = count // scan_height
                    data[row_start:row_start + row_count, :, :, :] = batch.reshape(
                        row_count, scan_height, detector_height, detector_width
                    )
                else:
                    flat_indices = np.arange(start, start + count)
                    xs = flat_indices // scan_height
                    ys = flat_indices % scan_height
                    for local_idx, (x, y) in enumerate(zip(xs, ys)):
                        data[int(x), int(y), :, :] = batch[local_idx]

                print(f"[mib] wrote {start + count}/{nframes} frames -> {out_path}")

    return out_path


def arina_transform(srcfilepath,srcfilename,destfilepath,destfilename,flat_path):
    if py4DSTEM is None:
        raise ImportError("py4DSTEM is required for ARINA master.h5 conversion")
    # Dataset path definition
    # Folder
    # srcfilepath = r'E:\Experiment_Data\0813\0813-4dstem\8'

    # srcfilename = '8-2_FOV759nm_CL160mm_300_300_2ms_2.1mrad_165kx_001_master.h5'

    try:
        FOV, Camera_length, scan_width, scan_height, unit = parse_metadata_from_filename(srcfilename)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return

    # Filename
    filename = os.path.join(srcfilepath, srcfilename)

    try:
        inferred_scan_width, inferred_scan_height = infer_scan_shape(
            filename, guessed_width=scan_width, guessed_height=scan_height
        )
    except Exception as exc:
        print(f"ERROR while inferring scan shape: {exc}")
        return

    if (inferred_scan_width, inferred_scan_height) != (scan_width, scan_height):
        print(
            f"Scan shape inferred from data: {inferred_scan_width}x{inferred_scan_height} "
            f"(filename suggested {scan_width}x{scan_height})"
        )

    scan_width, scan_height = inferred_scan_width, inferred_scan_height

    # Check whether data is split into multiple chunk files
    base_prefix = srcfilename.replace('_master.h5', '')
    data_parts = sorted(
        f for f in os.listdir(srcfilepath)
        if f.startswith(base_prefix) and '_data_' in f and f.endswith('.h5')
    )

    # Flatfield definition

    # Folder
    # ff_file_path = r'E:\Experiment_Data\Flatfield'

    # Filename
    # flat_field = os.path.join(ff_file_path,'Flatfield_200kv_unbin_0801.tif')
    flat_field = os.path.join(flat_path)
    flatfield = load_flatfield(flat_field)


    # # Dataset loading
    # # 4D STEM scan dimension
    # scan_width, scan_height = 300, 300
    # scan_width = int(splitname[2])
    # scan_height = int(splitname[3])
    print(f'FOV: {FOV}, Camera length: {Camera_length}, scan_shape: {scan_width}x{scan_height}')
    if data_parts:
        print(f'Found {len(data_parts)} data chunks: example -> {data_parts[0]}')
    else:
        print('No split data chunks detected next to master file.')


    try:
        print(filename)
        # # With flatfield application
        dataset = py4DSTEM.import_file(
            filename,
            scan_width=scan_width,
            flatfield=flatfield,
            binfactor=1
        )
    

        # # Without flatfield application
        # dataset = py4DSTEM.import_file(filename,scan_width=scan_width)

        # defect points correction
        [Rx,Ry,Qx,Qy] = dataset.shape
        for i in range(Rx):
            for j in range(Ry):
                data = dataset[i,j]
                dataset.data[i,j] = defect_points_correction(data,flatfield)

        dataset.data = np.array(dataset.data, dtype = "float32")
        # print(np.max(dataset.data))
        # print(np.min(dataset.data))
        # print(np.sum(dataset.data)/scan_width/scan_height)

        # dataset.filter_hot_pixels(
        #       0.5,
        #       1)

        # dataset.get_dp_mean()
        # probe_radius_pixels, probe_qx0, probe_qy0 = dataset.get_probe_size(
        #     thresh_upper=0.8,
        #     thresh_lower=0.00001,
        #     plot=False
        # )

        # calibration 
        # R_pixel_size = splitname[1]

        # match = re.findall(r"\d*\.\d+|\d+", R_pixel_size)  # \d+ 表示匹配 1 个或多个连续数字
        # FOV = float(match[0])

        calibrateR = int(FOV) / scan_width

        # if 'nm' in R_pixel_size:
        #     unit = 'nm'
        #     calibrateR = int(FOV)/scan_width
        # elif 'weim' in R_pixel_size:
        #     unit = 'nm'
        #     calibrateR = int(FOV*1000)/scan_width
        # else:
        #     unit = 'pixel'
        #     calibrateR = 0.365
        dataset.calibration.set_R_pixel_size(calibrateR)
        dataset.calibration.set_R_pixel_units(unit)

        index_list = [[47,0.05041],[60,0.03884],[77,0.03189],[98,0.02648],[125,0.02077],[160,0.01703],[205,0.01268],[330,0.00790],[410,0.00671]]
        # Camera_length = int(splitname[2][2:-2])
        calibrateQ = next((bi for ai, bi in index_list if ai == Camera_length), None)
        # diffraction space pixel size calibration - from calibrated convergence angle
        dataset.calibration.set_Q_pixel_size(calibrateQ)
        dataset.calibration.set_Q_pixel_units('A^-1')
        dataset.calibration

        savename = os.path.join(destfilepath,destfilename)
        py4DSTEM.save(
            savename,
            dataset,
            'o'
        )
    except Exception as e:
        print(f"ERROR：{e}")  # 记录错误信息

    
    # dm.dmWriter('my_image.dm4', dataset.data, pixelSize=dataset.pixelsize)



def copy_master_h5_files(src_dir, dest_dir, flat_path):

    os.makedirs(dest_dir, exist_ok=True)

    if os.path.isfile(src_dir):
        # Single ARINA master file or Merlin MIB file selected
        root, file = os.path.dirname(src_dir), os.path.basename(src_dir)
        if file.lower().endswith(".mib"):
            out_path = merlin_mib_to_h5(root, file, dest_dir)
            print(f"[mib] converted {src_dir} -> {out_path}")
            messagebox.showinfo("Info","Completed!")
            return
        if not file.endswith('master.h5'):
            messagebox.showwarning("Warning", "Please choose a master.h5, mib, or a folder containing master.h5/mib files.")
            return
        if not flat_path:
            messagebox.showwarning("Warning", "flatfield_path is required for ARINA master.h5 conversion.")
            return
        arina_transform(root, file, dest_dir, file, flat_path)
    else:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                rel_path = os.path.relpath(root, src_dir)
                dest_subdir = os.path.join(dest_dir, rel_path)
                os.makedirs(dest_subdir, exist_ok=True)

                if file.lower().endswith(".mib"):
                    dest_file = build_merlin_h5_name(file)
                    dest_path = os.path.join(dest_subdir, dest_file)
                    if os.path.exists(dest_path):
                        print(f'{dest_path} has existed!')
                    else:
                        out_path = merlin_mib_to_h5(root, file, dest_subdir, dest_file)
                        print(f"[mib] converted {os.path.join(root, file)} -> {out_path}")

                elif file.endswith('master.h5'):
                    if not flat_path:
                        print(f"Skip ARINA file without flatfield_path: {os.path.join(root, file)}")
                        continue
                    if os.path.exists(os.path.join(dest_subdir,file)):
                        print(f'{os.path.join(dest_subdir,file)} has existed!')
                    else:
                        arina_transform(root,file,dest_subdir,file,flat_path)
    
    messagebox.showinfo("Info","Completed!")


def convert_path(src_path, dest_path, flat_path=""):
    """Non-GUI conversion entry point for scripts and command line use."""
    os.makedirs(dest_path, exist_ok=True)
    if os.path.isfile(src_path):
        root, file = os.path.dirname(src_path), os.path.basename(src_path)
        if file.lower().endswith(".mib"):
            return [merlin_mib_to_h5(root, file, dest_path)]
        if file.endswith("master.h5"):
            if not flat_path:
                raise ValueError("flat_path is required for ARINA master.h5 conversion")
            arina_transform(root, file, dest_path, file, flat_path)
            return [os.path.join(dest_path, file)]
        raise ValueError(f"Unsupported source file: {src_path}")

    outputs = []
    for root, _, files in os.walk(src_path):
        rel_path = os.path.relpath(root, src_path)
        dest_subdir = os.path.join(dest_path, rel_path)
        os.makedirs(dest_subdir, exist_ok=True)
        for file in files:
            if file.lower().endswith(".mib"):
                dest_file = build_merlin_h5_name(file)
                dest_file_path = os.path.join(dest_subdir, dest_file)
                if os.path.exists(dest_file_path):
                    print(f"{dest_file_path} has existed!")
                    outputs.append(dest_file_path)
                else:
                    outputs.append(merlin_mib_to_h5(root, file, dest_subdir, dest_file))
            elif file.endswith("master.h5"):
                if not flat_path:
                    print(f"Skip ARINA file without flat_path: {os.path.join(root, file)}")
                    continue
                dest_file_path = os.path.join(dest_subdir, file)
                if os.path.exists(dest_file_path):
                    print(f"{dest_file_path} has existed!")
                    outputs.append(dest_file_path)
                else:
                    arina_transform(root, file, dest_subdir, file, flat_path)
                    outputs.append(dest_file_path)
    return outputs


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Convert ARINA master.h5 or Merlin mib 4D-STEM data to HDF5")
    parser.add_argument("--src", help="Source master.h5/mib file or folder")
    parser.add_argument("--dest", help="Destination folder")
    parser.add_argument("--flatfield", default="", help="Flatfield file for ARINA master.h5 conversion")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Tkinter GUI instead of command line conversion",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.gui or not (args.src and args.dest):
        root = tk.Tk()
        app = PathInputApp(root)
        root.mainloop()
    else:
        converted = convert_path(args.src, args.dest, args.flatfield)
        print(f"Completed. Converted {len(converted)} file(s).")

# if __name__ == "__main__":

#     src_path = r'E:\LFC'
#     dest_path = r'E:\Experiment_Data\12'
    
#     copy_master_h5_files(src_path, dest_path)
