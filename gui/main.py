import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

# WAŻNE: załaduj filtry, żeby się zarejestrowały w registry (efekt uboczny importu)
from glitchlab import filters as _filters  # noqa: F401

from glitchlab.core.pipeline import load_image, save_image, build_ctx, apply_pipeline, load_config

APP_TITLE = "GlitchLab — Preset Viewer"


def _to_rgb_uint8(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.ndim == 3 and a.shape[2] == 4:  # RGBA -> RGB na białym tle
        alpha = a[..., 3:4].astype(np.float32) / 255.0
        rgb = a[..., :3].astype(np.float32)
        a = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating):
            a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        else:
            a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def read_preset_yaml(preset_name: str):
    base = os.path.dirname(os.path.dirname(__file__))  # .../glitchlab
    path = os.path.join(base, "presets", f"{preset_name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return load_config(f.read())


def normalize_cfg(cfg, preset_name: str):
    """
    Akceptuje oba formaty:
      A) {steps: [...], edge_mask: ..., amplitude: ...}
      B) {preset_name: {steps: [...], ...}} lub {jakas_nazwa: {...}}
    Zwraca dict z kluczami na właściwym poziomie.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Zły format pliku preset: oczekiwano obiektu YAML (dict).")

    # idealny przypadek – już na właściwym poziomie
    if ("steps" in cfg) or ("edge_mask" in cfg) or ("amplitude" in cfg):
        return cfg

    # nazwana sekcja zawiera właściwy preset
    if preset_name in cfg and isinstance(cfg[preset_name], dict):
        return cfg[preset_name]

    # plik ma pojedynczą sekcję o innej nazwie
    if len(cfg) == 1:
        only_val = next(iter(cfg.values()))
        if isinstance(only_val, dict):
            return only_val

    raise ValueError(
        f"Nie znaleziono sekcji preset (steps/edge_mask/amplitude). "
        f"Sprawdź strukturę YAML dla '{preset_name}'."
    )


def list_presets():
    base = os.path.dirname(os.path.dirname(__file__))
    pdir = os.path.join(base, "presets")
    return sorted(os.path.splitext(fn)[0] for fn in os.listdir(pdir) if fn.lower().endswith(".yaml"))


def np_to_tk_img(arr: np.ndarray, max_side=880):
    arr = _to_rgb_uint8(arr)
    pil = Image.fromarray(arr, "RGB")
    w, h = pil.size
    s = max_side / max(w, h)
    if s < 1.0:
        pil = pil.resize((int(w * s), int(h * s)), Image.BICUBIC)
    return ImageTk.PhotoImage(pil)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("960x720")

        self.arr = None
        self.ctx = None
        self.result = None

        # top bar
        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Button(top, text="Otwórz obraz…", command=self.on_open).pack(side=tk.LEFT)

        ttk.Label(top, text="Preset:").pack(side=tk.LEFT, padx=(12, 4))
        self.preset_var = tk.StringVar(self)
        presets = list_presets() or ["default"]
        self.preset_var.set(presets[0])
        ttk.Combobox(top, values=presets, textvariable=self.preset_var, state="readonly", width=24).pack(side=tk.LEFT)

        ttk.Label(top, text="Seed:").pack(side=tk.LEFT, padx=(12, 4))
        self.seed_var = tk.IntVar(self, 7)
        ttk.Entry(top, textvariable=self.seed_var, width=8).pack(side=tk.LEFT)

        ttk.Button(top, text="Zastosuj preset", command=self.on_apply).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(top, text="Zapisz wynik…", command=self.on_save).pack(side=tk.LEFT, padx=(12, 0))

        self.canvas = tk.Label(self, text="(brak obrazu)")
        self.canvas.pack(expand=True)

        self.status = tk.Label(self, text="Gotowy", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def set_status(self, txt: str):
        self.status.config(text=txt)
        self.status.update_idletasks()

    def show_image(self, arr):
        tkimg = np_to_tk_img(arr)
        self.canvas.configure(image=tkimg, text="")
        self.canvas.image = tkimg

    def on_open(self):
        path = filedialog.askopenfilename(
            title="Wybierz obraz",
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.webp"), ("All", "*.*")]
        )
        if not path:
            return
        try:
            a = load_image(path)          # (H,W,C)
            a = _to_rgb_uint8(a)          # sanityzacja
            self.arr = a
            self.ctx = build_ctx(self.arr, seed=int(self.seed_var.get()), cfg={})
            self.show_image(self.arr)
            self.set_status(f"Wczytano: {os.path.basename(path)}  |  {self.arr.shape[1]}×{self.arr.shape[0]}")
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def on_apply(self):
        if self.arr is None:
            messagebox.showwarning("Uwaga", "Najpierw wczytaj obraz.")
            return
        try:
            preset_name = self.preset_var.get()
            raw_cfg = read_preset_yaml(preset_name)
            cfg = normalize_cfg(raw_cfg, preset_name)

            # zbuduj ctx z tym samym cfg (edge_mask/amplitude itd.)
            self.ctx = build_ctx(self.arr, seed=int(self.seed_var.get()), cfg=cfg)

            steps = cfg.get("steps", [])
            if not steps:
                raise ValueError(f"Preset '{preset_name}' nie zawiera kroków ('steps').")

            self.set_status(f"Przetwarzanie… ({preset_name}), kroki: {len(steps)}")
            out = apply_pipeline(self.arr.copy(), self.ctx, steps)
            out = _to_rgb_uint8(out)
            self.result = out
            self.show_image(out)
            self.set_status(f"OK: '{preset_name}' | kroki: {len(steps)}")
        except Exception as e:
            messagebox.showerror("Błąd", str(e))
            self.set_status("Błąd")

    def on_save(self):
        if self.result is None:
            messagebox.showinfo("Info", "Brak wyniku do zapisania.")
            return
        out_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if out_path:
            try:
                save_image(_to_rgb_uint8(self.result), out_path)
                self.set_status(f"Zapisano: {os.path.basename(out_path)}")
            except Exception as e:
                messagebox.showerror("Błąd", str(e))


if __name__ == "__main__":
    print(APP_TITLE, "| Python", sys.version)
    App().mainloop()
