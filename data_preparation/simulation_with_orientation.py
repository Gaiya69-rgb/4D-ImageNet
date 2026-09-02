# This is the data gen for type cluster
import os
import pickle
import time
import abtem
import ase
import dask
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import affine_transform, gaussian_filter
import multiprocessing as mp

np.random.seed(1121)


# =========================
# Plane-index (hkl) helpers
# =========================
def _gcd3(a: int, b: int, c: int) -> int:
    import math
    return math.gcd(abs(a), math.gcd(abs(b), abs(c)))


def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Find rotation R that best maps P -> Q (least squares), assuming both are centered.
    P, Q: (N,3)
    Return R: (3,3)
    """
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # reflection fix
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return R


def _reciprocal_basis_from_cell_rows(cell_rows: np.ndarray) -> np.ndarray:
    """
    Compute reciprocal basis (a*, b*, c*) from direct basis (a,b,c) given as ROW vectors in Cartesian.
    Convention: a*路a=1, a*路b=0, ...
    Scale (2蟺) does NOT matter for direction matching, so we use the non-2蟺 crystallographic convention.
    Returns: (3,3) where rows are [a*, b*, c*] in Cartesian coordinates.
    """
    a = np.asarray(cell_rows[0], dtype=np.float64)
    b = np.asarray(cell_rows[1], dtype=np.float64)
    c = np.asarray(cell_rows[2], dtype=np.float64)
    V = float(np.dot(a, np.cross(b, c)))
    if abs(V) < 1e-18:
        # degenerate cell; fallback to identity-like
        return np.eye(3, dtype=np.float64)

    a_star = np.cross(b, c) / V
    b_star = np.cross(c, a) / V
    c_star = np.cross(a, b) / V
    return np.vstack([a_star, b_star, c_star])  # rows


def _best_plane_hkl_from_normal(n_cart: np.ndarray, recip_rows: np.ndarray, max_index: int = 6) -> np.ndarray:
    """
    Given a target plane normal direction n_cart in Cartesian, find integer (h,k,l) such that
    g = h*a* + k*b* + l*c* (Cartesian) is most parallel to n_cart.
    recip_rows: rows are a*, b*, c* in Cartesian.
    """
    n = np.asarray(n_cart, dtype=np.float64)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return np.array([0, 0, 1], dtype=np.int16)
    n = n / nn

    a_star, b_star, c_star = recip_rows[0], recip_rows[1], recip_rows[2]

    best = None
    best_score = -1.0

    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for l in range(-max_index, max_index + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                g = _gcd3(h, k, l)
                if g != 1:
                    continue  # only primitive planes (avoid multiples)

                gvec = h * a_star + k * b_star + l * c_star
                gn = np.linalg.norm(gvec)
                if gn < 1e-12:
                    continue
                ghat = gvec / gn

                cos = float(np.dot(ghat, n))
                score = abs(cos)  # (hkl) and (-h-k-l) are same plane family
                if score > best_score:
                    best_score = score
                    if cos < 0:
                        best = np.array([-h, -k, -l], dtype=np.int16)
                    else:
                        best = np.array([h, k, l], dtype=np.int16)

    if best is None:
        best = np.array([0, 0, 1], dtype=np.int16)

    # canonicalize sign: first non-zero positive
    for i in range(3):
        if best[i] != 0:
            if best[i] < 0:
                best = (-best).astype(np.int16)
            break

    return best.astype(np.int16)


def main(dataset_folder, cif_folder, cif_name, cif_index, cif_count, flag_check, simulation_count):
    # global settings
    abtem.config.set({"visualize.cmap": "viridis"})
    abtem.config.set({"visualize.continuous_update": True})
    abtem.config.set({"visualize.autoscale": True})
    abtem.config.set({"visualize.reciprocal_space_units": "mrad"})
    abtem.config.set({"device": "gpu"})
    abtem.config.set({"fft": "fftw"})
    abtem.config.set({"dask.chunk-size-gpu": "8192 MB"})
    abtem.config.set({"dask.lazy": False})
    abtem.config.set({"cupy.fft-cache-size": "2048 MB"})
    dask.config.set({"num_workers": 1})

    config = {
        "sample_width": 150,
        "sample_width_sampling": 0.1,
        "sample_thickness": 100,
        "sample_thickness_sampling": 100,
        "target_size": 256,
        "semi_angle_cutoff_range": [2.1, 2.1],
        "max_angle_range": [41.9, 41.9],
        # 40.5 -> 0.01273202614379072
        # 41.4 -> 0.01299346405228745
        # 42.5 -> 0.013359477124182869
        # 53.1 -> 0.016653594771242213
        # 54.1 -> 0.0169673202614383
        # 55.0 -> 0.01722875816993504
        "sigma_heat_range": [0.0, 0.1],
        "sigma_blur_range": [0.5, 1.5],
        "translation_range": [0, 0],
    }

    # You can tune this range: larger => closer to true hkl but slower search
    HKL_MAX_INDEX = 6

    if not os.path.exists(f"{dataset_folder}"):
        os.makedirs(f"{dataset_folder}")

    # ---- read CIF (STRICT cell basis comes from CIF) ----
    primitive_unit = ase.io.read(os.path.join(cif_folder, cif_name))
    cif_cell_rows = np.array(primitive_unit.cell.array, dtype=np.float64)  # rows: a,b,c in Cartesian

    # ---- build orthogonalized sample for abTEM ----
    orthogonal_unit = abtem.orthogonalize_cell(primitive_unit)
    sample_cell_parameter_list = orthogonal_unit.cell.cellpar()
    sample_cell_length_min = np.min(sample_cell_parameter_list[0:3])
    sample_cell_count = int(
        np.ceil(np.max([config["sample_width"], config["sample_thickness"]]) / sample_cell_length_min * 2)
    )
    sample = orthogonal_unit * (sample_cell_count, sample_cell_count, sample_cell_count)
    sample.center(about=(config["sample_width"] / 2, config["sample_width"] / 2, config["sample_thickness"] / 2))

    probability_map_list = []
    center_list = []
    pixel_size_list = []

    # NEW: save both Euler angles (deg) and strict plane index (hkl)
    orientation_list = []  # (phi, theta, psi) in degrees
    hkl_list = []          # plane index (h, k, l) computed in CIF basis

    for i in range(simulation_count):
        time_start = time.time()

        probability_map, center, pixel_size, euler_deg, hkl = simulate(
            sample.copy(),
            config,
            cif_cell_rows=cif_cell_rows,
            hkl_max_index=HKL_MAX_INDEX
        )

        probability_map_list.append(probability_map)
        center_list.append(center)
        pixel_size_list.append(pixel_size)
        orientation_list.append(euler_deg)
        hkl_list.append(hkl)

        if flag_check:
            print([center, pixel_size, euler_deg, hkl])
            probability_map_preview = probability_map.copy()
            thr = np.quantile(probability_map_preview, 0.995)
            probability_map_preview[probability_map_preview > thr] = thr
            plt.imshow(probability_map_preview)
            plt.show()

        print(
            f"{(cif_index + (i + 1) / simulation_count) / cif_count * 100:.2f}% "
            f"{cif_name}_{i} completed in {time.time() - time_start:.2f} s"
        )

    orientation_list = np.asarray(orientation_list, dtype=np.float32)  # (N,3)
    hkl_list = np.asarray(hkl_list, dtype=np.int16)                    # (N,3)

    material = os.path.splitext(os.path.basename(cif_name))[0]
    diffraction_patterns = np.asarray(probability_map_list, dtype=np.float32)
    center_list = np.asarray(center_list, dtype=np.float32)
    pixel_size_list = np.asarray(pixel_size_list, dtype=np.float32)

    result = {
        "data_type": "simulation",
        "material": material,
        "source_cif": cif_name,
        "count": int(simulation_count),
        "diffraction_patterns": diffraction_patterns,
        "center_list": center_list,
        "pixel_size_list": pixel_size_list,
        "orientation_list": orientation_list,  # degrees: [phi, theta, psi]
        "hkl_list": hkl_list,                  # strict plane index (h,k,l) in CIF basis
        "simulation_parameters": config,
    }

    output_name = (
        f"{material}_sim_FOVnone_size{config['target_size']}"
        f"_thick{config['sample_thickness']}A"
        f"_conv{config['semi_angle_cutoff_range'][0]}mrad"
        f"_randomOrientation.pkl"
    )
    pickle.dump(result, open(os.path.join(dataset_folder, output_name), "wb"))


def simulate(sample, config, cif_cell_rows: np.ndarray, hkl_max_index: int = 6):
    sample_width = config["sample_width"]
    sample_width_sampling = config["sample_width_sampling"]
    sample_thickness = config["sample_thickness"]
    sample_thickness_sampling = config["sample_thickness_sampling"]
    target_size = config["target_size"]

    semi_angle_cutoff_range = config["semi_angle_cutoff_range"]
    max_angle_range = config["max_angle_range"]
    sigma_heat_range = config["sigma_heat_range"]
    sigma_blur_range = config["sigma_blur_range"]
    translation_range = config["translation_range"]

    semi_angle_cutoff = np.random.uniform(semi_angle_cutoff_range[0], semi_angle_cutoff_range[1])
    max_angle = np.random.uniform(max_angle_range[0], max_angle_range[1])
    sigma_heat = np.random.uniform(sigma_heat_range[0], sigma_heat_range[1])
    sigma_blur = np.random.uniform(sigma_blur_range[0], sigma_blur_range[1])

    # =======================
    # NEW: infer R and strict (hkl) plane index using CIF cell
    # =======================
    rot_center = np.array([sample_width / 2, sample_width / 2, sample_thickness / 2], dtype=np.float64)

    n_atoms = len(sample)
    if n_atoms >= 256:
        idx = np.random.choice(n_atoms, size=256, replace=False)
    else:
        idx = np.arange(n_atoms)

    pos0 = np.asarray(sample.positions[idx], dtype=np.float64).copy()

    # random Euler angles (deg) 鈥?your original logic
    phi = float(np.random.uniform(0, 360))
    theta = float(np.random.uniform(0, 180))
    psi = float(np.random.uniform(0, 360))
    sample.euler_rotate(phi=phi, theta=theta, psi=psi,
                        center=(sample_width / 2, sample_width / 2, sample_thickness / 2))
    euler_deg = (phi, theta, psi)

    pos1 = np.asarray(sample.positions[idx], dtype=np.float64).copy()

    P = pos0 - rot_center
    Q = pos1 - rot_center
    R = _kabsch_rotation(P, Q)

    # beam direction in lab (assume +z)
    z_lab = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # beam direction expressed in crystal cartesian BEFORE rotation:
    # If R maps crystal->lab, then n_crystal = R^T * z_lab
    n_crystal_cart = R.T @ z_lab

    # Reciprocal basis strictly from CIF cell (not orthogonalized cell)
    recip_rows = _reciprocal_basis_from_cell_rows(cif_cell_rows)

    # Find best plane index (hkl): g(hkl) || n_crystal_cart
    hkl = _best_plane_hkl_from_normal(n_crystal_cart, recip_rows, max_index=hkl_max_index)
    # =======================

    # cut cell
    sample.cell = (sample_width, sample_width, sample_thickness)
    sample = abtem.atoms.atoms_in_cell(sample, margin=0)

    # size fix
    expand_size = np.max(np.abs(translation_range))
    final_size = target_size + 2 * expand_size
    final_size = target_size
    max_angle = max_angle * final_size / target_size

    frozen_phonons = abtem.FrozenPhonons(
        sample,
        sigmas=sigma_heat,
        num_configs=8,
        seed=np.random.randint(1e6, int(1e9))
    )

    frozen_phonon_potential = abtem.Potential(
        frozen_phonons,
        sampling=sample_width_sampling,
        slice_thickness=sample_thickness_sampling
    )

    probe = abtem.Probe(
        energy=200e3,
        semiangle_cutoff=semi_angle_cutoff,
        sampling=sample_width_sampling
    )

    detector = abtem.PixelatedDetector(max_angle=max_angle)

    scan = abtem.GridScan(
        start=(sample_width / 2, sample_width / 2),
        end=(sample_width / 2 + 1, sample_width / 2 + 1),
        sampling=2,
        potential=frozen_phonon_potential
    )

    measurement_4d = probe.scan(scan=scan, potential=frozen_phonon_potential, detectors=detector)
    measurement_4d.compute(progress_bar=False, scheduler="threads", num_workers=1)

    reciprocal_x_coordinate_list = measurement_4d.axes_metadata[2].coordinates(measurement_4d.shape[2])
    reciprocal_y_coordinate_list = measurement_4d.axes_metadata[3].coordinates(measurement_4d.shape[3])
    pixel_size = [
        abs(reciprocal_x_coordinate_list[1] - reciprocal_x_coordinate_list[0]),
        abs(reciprocal_y_coordinate_list[1] - reciprocal_y_coordinate_list[0]),
    ]
    origin_x = np.argmin(np.abs(reciprocal_x_coordinate_list))
    origin_y = np.argmin(np.abs(reciprocal_y_coordinate_list))

    probability_map = measurement_4d[0].array[0, :, :]

    transformation_matrix = np.array([
        [1, 0, origin_x],
        [0, 1, origin_y],
        [0, 0, 1]
    ]) @ np.array([
        [1, 0, -(float(probability_map.shape[0]) - 1) / 2],
        [0, 1, -(float(probability_map.shape[1]) - 1) / 2],
        [0, 0, 1]
    ])
    probability_map = affine_transform(probability_map, transformation_matrix, order=1)

    pixel_size = [
        pixel_size[0] * (probability_map.shape[0] - 1) / (final_size - 1),
        pixel_size[1] * (probability_map.shape[1] - 1) / (final_size - 1),
    ]
    print(pixel_size)

    transformation_matrix = np.array([
        [1, 0, (float(probability_map.shape[0]) - 1) / 2],
        [0, 1, (float(probability_map.shape[1]) - 1) / 2],
        [0, 0, 1]
    ]) @ np.array([
        [(probability_map.shape[0] - 1) / (final_size - 1), 0, 0],
        [0, (probability_map.shape[1] - 1) / (final_size - 1), 0],
        [0, 0, 1]
    ]) @ np.array([
        [1, 0, -(float(final_size) - 1) / 2],
        [0, 1, -(float(final_size) - 1) / 2],
        [0, 0, 1]
    ])

    probability_map = affine_transform(probability_map, transformation_matrix, order=1)
    probability_map = gaussian_filter(probability_map, sigma_blur, truncate=10)

    probability_map = probability_map[
        0 + expand_size:final_size - expand_size,
        0 + expand_size:final_size - expand_size
    ]

    probability_map = probability_map / (np.sum(probability_map) + 1e-12)
    center = [(float(target_size) - 1) / 2, (float(target_size) - 1) / 2]

    # NEW: return Euler + strict plane (hkl)
    return probability_map, center, pixel_size, euler_deg, hkl


def process(file_name_list, dataset_folder, cif_folder, flag_check, simulation_count):
    cif_count = len(file_name_list)
    for cif_index in range(cif_count):
        cif_name = file_name_list[cif_index]
        if not cif_name.endswith(".cif"):
            continue
        main(dataset_folder, cif_folder, cif_name, cif_index, cif_count, flag_check, simulation_count)


if __name__ == '__main__':
    OBJECT_NAME = rf'Ag'
    dataset_folder = rf"G:\4D-ImageNet\sim\{OBJECT_NAME}"
    cif_folder = rf"G:\4D-ImageNet\cif"

    process_count = 1
    cif_count_per_process = 1
    simulation_count = 2000
    file_name_list = [f"{OBJECT_NAME}.cif"]
    print(file_name_list)
    flag_check = False

    thread_list = []
    for i in range(process_count):
        thread_list.append(
            mp.Process(
                target=process,
                args=(
                    file_name_list[i * cif_count_per_process:(i + 1) * cif_count_per_process],
                    dataset_folder, cif_folder, flag_check, simulation_count
                )
            )
        )
    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()


