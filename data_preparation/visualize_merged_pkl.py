import importlib
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import numpy as np


class NumpyCompatUnpickler(pickle.Unpickler):
    MODULE_CANDIDATES = {
        ("numpy._core.numeric", "_frombuffer"): [
            ("numpy._core.numeric", "_frombuffer"),
            ("numpy.core.numeric", "_frombuffer"),
        ],
        ("numpy._core.multiarray", "_reconstruct"): [
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "_reconstruct"),
        ],
        ("numpy.core.numeric", "_frombuffer"): [
            ("numpy._core.numeric", "_frombuffer"),
            ("numpy.core.numeric", "_frombuffer"),
        ],
        ("numpy.core.multiarray", "_reconstruct"): [
            ("numpy._core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "_reconstruct"),
        ],
    }

    def find_class(self, module, name):
        candidates = self.MODULE_CANDIDATES.get((module, name))
        if candidates is not None:
            for candidate_module, candidate_name in candidates:
                try:
                    imported = importlib.import_module(candidate_module)
                    getattr(imported, candidate_name)
                    module, name = candidate_module, candidate_name
                    break
                except (ImportError, AttributeError):
                    continue
        return super().find_class(module, name)


def choose_pkl_file():
    root = tk.Tk()
    root.withdraw()
    root.update()
    file_path = filedialog.askopenfilename(
        title="Choose a merged pkl file",
        filetypes=[("PKL files", "*.pkl"), ("All files", "*.*")],
    )
    root.destroy()
    return file_path


def load_merged_payload(file_path):
    with open(file_path, "rb") as f:
        payload = NumpyCompatUnpickler(f).load()

    if not isinstance(payload, dict):
        raise ValueError("PKL content is not a dict")
    if "diffraction_patterns" not in payload:
        raise KeyError("This does not look like a merged diffraction pkl")

    patterns = np.asarray(payload["diffraction_patterns"])
    scan_positions = payload.get("scan_positions")
    if scan_positions is not None:
        scan_positions = np.asarray(scan_positions)

    if patterns.ndim != 3:
        raise ValueError(f"Expected 3D diffraction_patterns, got shape {patterns.shape}")
    if scan_positions is not None:
        if scan_positions.ndim != 2 or scan_positions.shape[1] != 2:
            raise ValueError(f"Expected scan_positions shape (N, 2), got {scan_positions.shape}")
        if patterns.shape[0] != scan_positions.shape[0]:
            raise ValueError("Pattern count does not match scan_positions count")

    return payload, patterns, scan_positions


def build_title(payload, file_path, index, scan_position, score):
    source_name = payload.get("source_name") or payload.get("source_cif", "unknown")
    total = int(payload.get("count", 0)) or len(payload.get("diffraction_patterns", []))
    data_type = payload.get("data_type", "experiment")

    if data_type == "simulation":
        title_parts = [source_name, file_path, f"index {index + 1}/{total}"]
        orientation_list = payload.get("orientation_list")
        hkl_list = payload.get("hkl_list")
        if orientation_list is not None:
            phi, theta, psi = np.asarray(orientation_list)[index]
            title_parts.append(f"Euler phi={phi:.2f}, theta={theta:.2f}, psi={psi:.2f} deg")
        if hkl_list is not None:
            h, k, l = np.asarray(hkl_list)[index]
            title_parts.append(f"hkl=({int(h)}, {int(k)}, {int(l)})")
        return "\n".join(title_parts)

    pos_x = int(scan_position[0]) if scan_position is not None else "?"
    pos_y = int(scan_position[1]) if scan_position is not None else "?"
    return (
        f"{source_name}\n"
        f"{file_path}\n"
        f"index {index + 1}/{total}, scan x={pos_x}, y={pos_y}, score={score:.4f}"
    )


def show_merged_patterns(payload, patterns, scan_positions, file_path):
    scores = np.asarray(payload.get("scores", np.zeros(patterns.shape[0], dtype=np.float32)))
    current_index = 0

    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.18)

    image = ax.imshow(patterns[current_index], cmap="gray")
    fig.colorbar(image, ax=ax, label="Intensity")
    ax.axis("off")

    def redraw(index):
        nonlocal current_index
        current_index = max(0, min(index, patterns.shape[0] - 1))
        image.set_data(patterns[current_index])
        image.set_clim(vmin=float(patterns[current_index].min()), vmax=float(patterns[current_index].max()))
        ax.set_title(
            build_title(
                payload,
                file_path,
                current_index,
                scan_positions[current_index] if scan_positions is not None else None,
                float(scores[current_index]),
            )
        )
        index_box.set_val(str(current_index))
        fig.canvas.draw_idle()

    def go_prev(_event):
        redraw(current_index - 1)

    def go_next(_event):
        redraw(current_index + 1)

    def submit_index(text):
        try:
            redraw(int(text))
        except ValueError:
            redraw(current_index)

    prev_ax = plt.axes([0.22, 0.05, 0.12, 0.06])
    next_ax = plt.axes([0.36, 0.05, 0.12, 0.06])
    index_ax = plt.axes([0.55, 0.05, 0.20, 0.06])

    prev_button = Button(prev_ax, "Prev")
    next_button = Button(next_ax, "Next")
    index_box = TextBox(index_ax, "Index ", initial="0")

    prev_button.on_clicked(go_prev)
    next_button.on_clicked(go_next)
    index_box.on_submit(submit_index)

    redraw(0)
    plt.show()


def main():
    file_path = choose_pkl_file()
    if not file_path:
        return

    try:
        payload, patterns, scan_positions = load_merged_payload(file_path)
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", str(exc))
        root.destroy()
        return

    show_merged_patterns(payload, patterns, scan_positions, file_path)


if __name__ == "__main__":
    main()


