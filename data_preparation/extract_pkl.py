#Libraries import

import numpy as np

import tifffile
import os
import math
import heapq
import pickle
import argparse

import tkinter as tk
from tkinter import filedialog, messagebox
import re
import pathlib

try:
    import h5py
except ImportError:  # pragma: no cover - h5py expected in py4DSTEM env
    h5py = None

try:
    import py4DSTEM
except ImportError:  # pragma: no cover - optional for extraction workflow
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
        # Allow selecting either a folder (batch) or a single master.h5 file
        path = filedialog.askopenfilename(
            title="Choose src folder or master.h5",
            filetypes=[("HDF5/NeXus", "*.h5"), ("All files", "*.*")])
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

        if not src_path or not dest_path or not flat_path:
            messagebox.showwarning("Warning", "Please fill in every path!")
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


def get_datacube_dataset(h5_file):
    """Return the raw 4D datacube dataset stored in a master h5 file."""
    candidates = [
        ("datacube_root", "datacube", "data"),
        ("entry", "data", "data"),
    ]
    for path in candidates:
        node = h5_file
        try:
            for key in path:
                node = node[key]
            if hasattr(node, "shape") and len(node.shape) == 4:
                return node
        except KeyError:
            continue

    found = []

    def visitor(name, obj):
        if hasattr(obj, "shape") and len(obj.shape) == 4:
            found.append((name, obj))

    h5_file.visititems(visitor)
    if not found:
        raise ValueError("Cannot find 4D datacube dataset in h5 file")
    return found[0][1]


def build_diffraction_masks(dp_shape, center_radius=12, outer_radius=20, beam_center=None):
    qx, qy = dp_shape
    if beam_center is None:
        cy, cx = qx / 2.0, qy / 2.0
    else:
        cy, cx = float(beam_center[0]), float(beam_center[1])
    y, x = np.ogrid[:qx, :qy]
    radius2 = (y - cy) ** 2 + (x - cx) ** 2
    center_mask = radius2 <= center_radius ** 2
    outer_mask = radius2 >= outer_radius ** 2
    non_center_mask = ~center_mask
    return center_mask, outer_mask, non_center_mask


def estimate_beam_center(dataset, max_samples=256):
    """Estimate the direct-beam center from a sparse scan-grid average."""
    scan_x, scan_y, qx, qy = dataset.shape
    sample_count = min(max_samples, scan_x * scan_y)
    grid_x = max(1, int(round((scan_x * scan_y / sample_count) ** 0.5)))
    step_x = max(1, scan_x // grid_x)
    step_y = max(1, scan_y // grid_x)

    accum = np.zeros((qx, qy), dtype=np.float64)
    count = 0
    for ix in range(0, scan_x, step_x):
        for iy in range(0, scan_y, step_y):
            accum += np.asarray(dataset[ix, iy], dtype=np.float64)
            count += 1
            if count >= sample_count:
                break
        if count >= sample_count:
            break

    if count == 0:
        return qx / 2.0, qy / 2.0

    mean_pattern = accum / count
    total = float(mean_pattern.sum())
    if total <= 0:
        return qx / 2.0, qy / 2.0

    yy, xx = np.indices(mean_pattern.shape, dtype=np.float64)
    center_y = float((yy * mean_pattern).sum() / total)
    center_x = float((xx * mean_pattern).sum() / total)
    return center_y, center_x


def compute_diffraction_score(dp, center_mask, outer_mask, non_center_mask, beam_center=None):
    arr = np.asarray(dp, dtype=np.float32)
    total = float(arr.sum())
    if total <= 0:
        return -1.0, {
            "total_intensity": total,
            "center_fraction": 1.0,
            "outer_fraction": 0.0,
            "non_center_std": 0.0,
            "non_center_top1_fraction": 0.0,
            "beam_center_distance": float("inf"),
        }

    center_sum = float(arr[center_mask].sum())
    outer_sum = float(arr[outer_mask].sum())
    non_center = arr[non_center_mask]
    non_center_std = float(non_center.std())
    top1_fraction = float(non_center.max() / total) if non_center.size else 0.0
    center_fraction = center_sum / total
    outer_fraction = outer_sum / total
    yy, xx = np.indices(arr.shape, dtype=np.float32)
    com_y = float((yy * arr).sum() / total)
    com_x = float((xx * arr).sum() / total)
    if beam_center is None:
        beam_center = (arr.shape[0] / 2.0, arr.shape[1] / 2.0)
    beam_center_distance = float(
        ((com_y - float(beam_center[0])) ** 2 + (com_x - float(beam_center[1])) ** 2) ** 0.5
    )

    # Prefer patterns that still retain a recognizable central beam,
    # but are not dominated by only the center spot.
    target_center_fraction = 0.22
    center_fraction_tolerance = 0.18
    center_balance = max(
        0.0,
        1.0 - abs(center_fraction - target_center_fraction) / center_fraction_tolerance,
    )
    beam_alignment = max(0.0, 1.0 - beam_center_distance / 12.0)

    score = (
        0.35 * center_balance +
        0.35 * outer_fraction +
        0.20 * beam_alignment +
        0.10 * min(non_center_std / max(arr.mean(), 1e-6), 10.0) / 10.0
    )

    metrics = {
        "total_intensity": total,
        "center_fraction": center_fraction,
        "outer_fraction": outer_fraction,
        "non_center_std": non_center_std,
        "non_center_top1_fraction": top1_fraction,
        "beam_center_distance": beam_center_distance,
        "center_of_mass": {"y": com_y, "x": com_x},
    }
    return float(score), metrics


def select_top_diffraction_patterns(
    dataset,
    top_k=5000,
    center_radius=12,
    outer_radius=20,
    beam_center=None,
):
    scan_x, scan_y, qx, qy = dataset.shape
    center_mask, outer_mask, non_center_mask = build_diffraction_masks(
        (qx, qy),
        center_radius=center_radius,
        outer_radius=outer_radius,
        beam_center=beam_center,
    )
    heap = []
    total_frames = scan_x * scan_y

    for ix in range(scan_x):
        for iy in range(scan_y):
            dp = dataset[ix, iy]
            score, metrics = compute_diffraction_score(
                dp,
                center_mask,
                outer_mask,
                non_center_mask,
                beam_center=beam_center,
            )
            item = (score, int(ix), int(iy), metrics)
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif score > heap[0][0]:
                heapq.heapreplace(heap, item)

    selected = sorted(heap, key=lambda item: item[0], reverse=True)
    return selected, total_frames


def sanitize_stem(name):
    return re.sub(r"[^\w\-.]+", "_", name)


def save_selected_patterns(
    master_path,
    output_dir,
    top_k=5000,
    center_radius=12,
    outer_radius=20,
    overwrite=False,
):
    os.makedirs(output_dir, exist_ok=True)
    base_name = pathlib.Path(master_path).stem
    metadata = parse_metadata_from_filename(os.path.basename(master_path))
    fov, camera_length, scan_width, scan_height, unit = metadata

    existing_pkls = list(pathlib.Path(output_dir).glob("*.pkl"))
    if len(existing_pkls) >= top_k and not overwrite:
        return {
            "master_path": os.path.abspath(master_path),
            "output_dir": os.path.abspath(output_dir),
            "selected_count": len(existing_pkls),
            "available_count": None,
            "best_score": None,
            "worst_score": None,
            "skipped_existing": True,
        }

    with h5py.File(master_path, "r") as h5_file:
        dataset = get_datacube_dataset(h5_file)
        beam_center = estimate_beam_center(dataset)
        target_k = min(top_k, dataset.shape[0] * dataset.shape[1])
        if overwrite:
            for pkl_path in pathlib.Path(output_dir).glob("*.pkl"):
                pkl_path.unlink()
        selected, total_frames = select_top_diffraction_patterns(
            dataset,
            top_k=target_k,
            center_radius=center_radius,
            outer_radius=outer_radius,
            beam_center=beam_center,
        )
        for rank, (score, ix, iy, metrics) in enumerate(selected):
            dp = np.asarray(dataset[ix, iy], dtype=np.float32)
            payload = {
                "source_file": os.path.abspath(master_path),
                "source_name": os.path.basename(master_path),
                "material": pathlib.Path(master_path).parent.name,
                "scan_position": {"x": int(ix), "y": int(iy)},
                "scan_shape": {"x": int(dataset.shape[0]), "y": int(dataset.shape[1])},
                "diffraction_shape": {"qx": int(dataset.shape[2]), "qy": int(dataset.shape[3])},
                "fov": fov,
                "camera_length": camera_length,
                "real_space_unit": unit,
                "filename_scan_shape": {"x": scan_width, "y": scan_height},
                "estimated_beam_center": {"y": float(beam_center[0]), "x": float(beam_center[1])},
                "score": float(score),
                "selection_rank": int(rank),
                "selection_metrics": metrics,
                "diffraction_pattern": dp,
            }
            output_name = f"{rank:04d}_x{ix:03d}_y{iy:03d}.pkl"
            output_path = os.path.join(output_dir, output_name)
            with open(output_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    kept = len(selected)
    return {
        "master_path": os.path.abspath(master_path),
        "output_dir": os.path.abspath(output_dir),
        "selected_count": kept,
        "available_count": total_frames,
        "beam_center": beam_center,
        "best_score": float(selected[0][0]) if selected else None,
        "worst_score": float(selected[-1][0]) if selected else None,
    }


def extract_patterns_from_workspace(
    src_root,
    output_root=None,
    top_k=5000,
    center_radius=12,
    outer_radius=20,
    overwrite=False,
):
    if h5py is None:
        raise ImportError("h5py is required for diffraction pattern extraction")

    src_root = os.path.abspath(src_root)
    if output_root is None:
        output_root = src_root
    else:
        output_root = os.path.abspath(output_root)

    src_path = pathlib.Path(src_root)
    if src_path.is_file():
        master_files = [src_path]
    else:
        master_files = [
            path for path in sorted(src_path.glob("*/*_master.h5"))
            if path.parent.name.lower() != "flatfield" and not path.name.startswith("._")
        ]
    if not master_files:
        raise FileNotFoundError(f"No *_master.h5 files found under {src_root}")

    summaries = []
    for master_path in master_files:
        relative_parent = pathlib.Path(".") if src_path.is_file() else master_path.parent.relative_to(src_root)
        target_dir = pathlib.Path(output_root) / relative_parent / sanitize_stem(master_path.stem)
        print(f"[extract] {master_path} -> {target_dir}")
        summary = save_selected_patterns(
            str(master_path),
            str(target_dir),
            top_k=top_k,
            center_radius=center_radius,
            outer_radius=outer_radius,
            overwrite=overwrite,
        )
        summaries.append(summary)
        if summary.get("skipped_existing"):
            print(f"[skip] found {summary['selected_count']} existing pkl files")
        else:
            print(
                f"[done] kept {summary['selected_count']}/{summary['available_count']} "
                f"patterns, beam center ({summary['beam_center'][0]:.2f}, {summary['beam_center'][1]:.2f}), "
                f"score range {summary['worst_score']:.4f} - {summary['best_score']:.4f}"
            )
    return summaries


def build_arg_parser():
    parser = argparse.ArgumentParser(description="ARINA 4D-STEM tools")
    parser.add_argument(
        "--extract-patterns",
        action="store_true",
        help="Extract top diffraction patterns from each *_master.h5 into per-file pkl folders.",
    )
    parser.add_argument(
        "--src-root",
        default=os.getcwd(),
        help="Workspace root containing material subfolders.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Where extracted pkl folders should be written. Defaults to src-root.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5000,
        help="Number of diffraction patterns to keep per master file.",
    )
    parser.add_argument(
        "--center-radius",
        type=int,
        default=12,
        help="Radius in pixels used to estimate center-spot dominance.",
    )
    parser.add_argument(
        "--outer-radius",
        type=int,
        default=20,
        help="Radius in pixels used to estimate outer diffraction content.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing pkl files in target folders.",
    )
    return parser


def arina_transform(srcfilepath,srcfilename,destfilepath,destfilename,flat_path):
    if py4DSTEM is None:
        raise ImportError("py4DSTEM is required for arina_transform")
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
        # Single master file selected
        root, file = os.path.dirname(src_dir), os.path.basename(src_dir)
        if not file.endswith('master.h5'):
            messagebox.showwarning("Warning", "Please choose a master.h5 or a folder containing master.h5 files.")
            return
        arina_transform(root, file, dest_dir, file, flat_path)
    else:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith('master.h5'):

                    rel_path = os.path.relpath(root, src_dir)
                    dest_subdir = os.path.join(dest_dir, rel_path)

                    os.makedirs(dest_subdir, exist_ok=True)
                    if os.path.exists(os.path.join(dest_subdir,file)):
                        print(f'{os.path.join(dest_subdir,file)} has existed!')
                    else:
                        arina_transform(root,file,dest_subdir,file,flat_path)
    
    messagebox.showinfo("Info","Completed!")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.extract_patterns:
        extract_patterns_from_workspace(
            src_root=args.src_root,
            output_root=args.output_root,
            top_k=args.top_k,
            center_radius=args.center_radius,
            outer_radius=args.outer_radius,
            overwrite=args.overwrite,
        )
    else:
        root = tk.Tk()
        app = PathInputApp(root)
        root.mainloop()

# if __name__ == "__main__":

#     src_path = r'E:\LFC'
#     dest_path = r'E:\Experiment_Data\12'
    
#     copy_master_h5_files(src_path, dest_path)
