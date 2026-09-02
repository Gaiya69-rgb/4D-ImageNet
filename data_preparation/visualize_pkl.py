import pickle
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np


class NumpyCompatUnpickler(pickle.Unpickler):
    """Map numpy internal module paths across numpy versions."""

    MODULE_ALIASES = {
        ("numpy._core.numeric", "_frombuffer"): ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.multiarray", "_reconstruct"): ("numpy.core.multiarray", "_reconstruct"),
    }

    def find_class(self, module, name):
        target = self.MODULE_ALIASES.get((module, name))
        if target is not None:
            module, name = target
        return super().find_class(module, name)


def choose_pkl_file():
    root = tk.Tk()
    root.withdraw()
    root.update()
    file_path = filedialog.askopenfilename(
        title="Choose a pkl file",
        filetypes=[("PKL files", "*.pkl"), ("All files", "*.*")],
    )
    root.destroy()
    return file_path


def load_diffraction_pattern(file_path):
    with open(file_path, "rb") as f:
        payload = NumpyCompatUnpickler(f).load()

    if not isinstance(payload, dict):
        raise ValueError("PKL content is not a dict")
    if "diffraction_pattern" not in payload:
        raise KeyError("Missing 'diffraction_pattern' in PKL file")

    pattern = np.asarray(payload["diffraction_pattern"])
    if pattern.ndim != 2:
        raise ValueError(f"Expected 2D diffraction pattern, got shape {pattern.shape}")

    return payload, pattern


def show_diffraction_pattern(payload, pattern, file_path):
    source_name = payload.get("source_name", "unknown")
    scan_position = payload.get("scan_position", {})
    pos_x = scan_position.get("x", "?")
    pos_y = scan_position.get("y", "?")
    beam_center = payload.get("estimated_beam_center")

    plt.figure(figsize=(7, 7))
    plt.imshow(pattern, cmap="gray")
    plt.colorbar(label="Intensity")
    if isinstance(beam_center, dict):
        center_y = beam_center.get("y")
        center_x = beam_center.get("x")
        if center_y is not None and center_x is not None:
            plt.axhline(center_y, color="red", linestyle="--", linewidth=0.8)
            plt.axvline(center_x, color="red", linestyle="--", linewidth=0.8)
    plt.title(f"{source_name}\n{file_path}\nscan x={pos_x}, y={pos_y}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    file_path = choose_pkl_file()
    if not file_path:
        return

    try:
        payload, pattern = load_diffraction_pattern(file_path)
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", str(exc))
        root.destroy()
        return

    show_diffraction_pattern(payload, pattern, file_path)


if __name__ == "__main__":
    main()
