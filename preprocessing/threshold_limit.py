"""
Threshold Selector — Tkinter, compatible Windows.
Usage : python threshold_app.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from pathlib import Path
import cv2
try:
    from PIL import Image, ImageTk, ImageFilter
    import numpy as np
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow", "numpy"])
    from PIL import Image, ImageTk, ImageFilter
    import numpy as np


# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 220
PREVIEW_SIZE      = 600
DEBOUNCE_MS       = 80

BG     = "#0f0f0f"
PANEL  = "#1a1a1a"
ACCENT = "#00e5ff"
TEXT   = "#e0e0e0"
MUTED  = "#555555"
BTN_BG = "#2a2a2a"
GREEN  = "#00e676"
ORANGE = "#ffb300"
RED    = "#ff5252"


# ─── Traitement image ─────────────────────────────────────────────────────────

def apply_threshold(arr: np.ndarray, threshold: int) -> np.ndarray:
    if arr.ndim == 3:
        gray = (0.299 * arr[:, :, 0] +
                0.587 * arr[:, :, 1] +
                0.114 * arr[:, :, 2]).astype(np.uint8)
    else:
        gray = arr.astype(np.uint8)
    return np.where(gray >= threshold, 255, 0).astype(np.uint8)


def post_process(mask: np.ndarray) -> np.ndarray:
    """Post-traitement : fermeture morphologique legere. Modifiez ici."""
    img = Image.fromarray(mask, mode="L")
    img = img.filter(ImageFilter.MaxFilter(5))
    img = img.filter(ImageFilter.MinFilter(5))
    img = np.array(img)  # conversion PIL → numpy
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    return np.array(img)


# ─── Application ──────────────────────────────────────────────────────────────

class ThresholdApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Threshold Selector")
        self.configure(bg=BG)

        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        win_w = min(920, sw - 80)
        win_h = min(860, sh - 80)
        self.geometry(f"{win_w}x{win_h}+{(sw-win_w)//2}+{(sh-win_h)//2}")
        self.minsize(700, 580)
        self.resizable(True, True)

        self._input_dir:   Path | None = None
        self._output_dir:  Path | None = None
        self._image_list:  list[Path] = []
        self._current_idx: int = 0
        self._current_arr: np.ndarray | None = None
        self._threshold:   int = DEFAULT_THRESHOLD
        self._debounce_id  = None
        self._loading      = False

        self._build_ui()

    # ── Construction UI ───────────────────────────────────────────────────────

    def _build_ui(self):

        # ── Barre HAUTE (hauteur fixe) ────────────────────────────────────────
        top = tk.Frame(self, bg=PANEL, height=46)
        top.pack(side=tk.TOP, fill=tk.X)
        top.pack_propagate(False)

        tk.Label(top, text="THRESHOLD SELECTOR",
                 font=("Courier New", 12, "bold"),
                 fg=ACCENT, bg=PANEL).pack(side=tk.LEFT, padx=14, pady=10)

        btn_kw = dict(font=("Courier New", 9), bg=BTN_BG, fg=TEXT,
                      activebackground=ACCENT, activeforeground="#000",
                      relief=tk.FLAT, padx=10, pady=4, cursor="hand2", bd=0)

        tk.Button(top, text="Dossier source",
                  command=self._pick_input, **btn_kw).pack(side=tk.LEFT, padx=4, pady=8)
        tk.Button(top, text="Dossier output",
                  command=self._pick_output, **btn_kw).pack(side=tk.LEFT, padx=4, pady=8)

        self._lbl_dirs = tk.Label(top, text="Aucun dossier selectionne",
                                  font=("Courier New", 8), fg=MUTED, bg=PANEL)
        self._lbl_dirs.pack(side=tk.LEFT, padx=10)

        # ── Barre BASSE fixe — slider + navigation + valider ──────────────────
        # Déclarée AVANT le canvas pour rester toujours visible (pack BOTTOM)
        bot = tk.Frame(self, bg=PANEL, height=118)
        bot.pack(side=tk.BOTTOM, fill=tk.X)
        bot.pack_propagate(False)

        # Ligne 1 : label "THRESHOLD" + valeur numérique
        row1 = tk.Frame(bot, bg=PANEL)
        row1.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(8, 0))

        tk.Label(row1, text="THRESHOLD",
                 font=("Courier New", 9), fg=MUTED, bg=PANEL).pack(side=tk.LEFT)

        self._lbl_val = tk.Label(row1, text=str(DEFAULT_THRESHOLD),
                                 font=("Courier New", 16, "bold"),
                                 fg=ACCENT, bg=PANEL, width=4, anchor="e")
        self._lbl_val.pack(side=tk.RIGHT)

        # Ligne 2 : le slider (tk.Scale natif, visible sous Windows)
        self._slider_var = tk.IntVar(value=DEFAULT_THRESHOLD)
        self._slider = tk.Scale(
            bot,
            from_=0, to=255,
            orient=tk.HORIZONTAL,
            variable=self._slider_var,
            command=self._on_slider,
            bg=PANEL,
            fg=ACCENT,
            troughcolor="#333333",
            activebackground=ACCENT,
            highlightthickness=0,
            sliderlength=24,
            width=14,
            showvalue=False,
            bd=0,
        )
        self._slider.pack(side=tk.TOP, fill=tk.X, padx=16)

        # Ligne 3 : min/max labels
        row3 = tk.Frame(bot, bg=PANEL)
        row3.pack(side=tk.TOP, fill=tk.X, padx=16)
        tk.Label(row3, text="0",   font=("Courier New", 8), fg=MUTED, bg=PANEL).pack(side=tk.LEFT)
        tk.Label(row3, text="255", font=("Courier New", 8), fg=MUTED, bg=PANEL).pack(side=tk.RIGHT)

        # Ligne 4 : navigation + status + bouton VALIDER
        row4 = tk.Frame(bot, bg=PANEL)
        row4.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(4, 8))

        nav_kw = dict(font=("Courier New", 9), bg=BTN_BG, fg=TEXT,
                      activebackground="#444", activeforeground=TEXT,
                      relief=tk.FLAT, padx=12, pady=4, cursor="hand2", bd=0)

        self._btn_prev = tk.Button(row4, text="< Prec.",
                                   command=self._prev_image,
                                   state=tk.DISABLED, **nav_kw)
        self._btn_prev.pack(side=tk.LEFT, padx=(0, 4))

        self._btn_next = tk.Button(row4, text="Suiv. >",
                                   command=self._next_image,
                                   state=tk.DISABLED, **nav_kw)
        self._btn_next.pack(side=tk.LEFT)

        self._lbl_status = tk.Label(row4, text="",
                                    font=("Courier New", 9), fg=GREEN, bg=PANEL)
        self._lbl_status.pack(side=tk.LEFT, padx=14)

        # Bouton VALIDER — fond cyan, texte noir, toujours visible
        self._btn_validate = tk.Button(
            row4,
            text="  VALIDER ET SAUVEGARDER  ",
            font=("Courier New", 10, "bold"),
            bg=ACCENT, fg="#000000",
            activebackground="#00b8d4", activeforeground="#000000",
            relief=tk.FLAT, padx=16, pady=5,
            cursor="hand2", bd=0,
            command=self._validate,
            state=tk.DISABLED,
        )
        self._btn_validate.pack(side=tk.RIGHT)

        # ── Zone IMAGE (espace restant au centre) ─────────────────────────────
        mid = tk.Frame(self, bg=BG)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Infos fichier (nom + compteur)
        info = tk.Frame(mid, bg=BG, height=22)
        info.pack(side=tk.TOP, fill=tk.X)
        info.pack_propagate(False)

        self._lbl_filename = tk.Label(info, text="—",
                                      font=("Courier New", 9, "bold"),
                                      fg=ACCENT, bg=BG)
        self._lbl_filename.pack(side=tk.LEFT, padx=4)

        self._lbl_counter = tk.Label(info, text="",
                                     font=("Courier New", 9), fg=MUTED, bg=BG)
        self._lbl_counter.pack(side=tk.RIGHT, padx=4)

        # Canvas
        self._canvas = tk.Canvas(mid, bg="#111111",
                                 highlightthickness=1,
                                 highlightbackground="#333333")
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._canvas.bind("<Configure>", self._on_resize)

        self._img_id    = None
        self._tk_img    = None
        self._ph_id     = self._canvas.create_text(
            400, 300,
            text="Selectionnez un dossier source pour commencer",
            font=("Courier New", 11), fill=MUTED, anchor="center"
        )

    # ── Dossiers ──────────────────────────────────────────────────────────────

    def _pick_input(self):
        d = filedialog.askdirectory(title="Dossier source")
        if not d:
            return
        self._input_dir = Path(d)
        self._load_list()
        self._refresh_dir_label()

    def _pick_output(self):
        d = filedialog.askdirectory(title="Dossier output")
        if not d:
            return
        self._output_dir = Path(d)
        self._refresh_dir_label()
        self._refresh_validate()

    def _refresh_dir_label(self):
        def t(s, n=38): return s if len(s) <= n else "..." + s[-(n-1):]
        src = t(str(self._input_dir))  if self._input_dir  else "—"
        out = t(str(self._output_dir)) if self._output_dir else "—"
        self._lbl_dirs.config(text=f"src: {src}   out: {out}")

    def _load_list(self):
        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
        self._image_list = sorted(
            p for p in self._input_dir.iterdir()
            if p.suffix.lower() in exts
        )
        self._current_idx = 0
        if not self._image_list:
            messagebox.showinfo("Info", "Aucune image trouvee dans ce dossier.")
            return
        if self._ph_id:
            self._canvas.delete(self._ph_id)
            self._ph_id = None
        self._load_current()

    def _refresh_validate(self):
        ok = (self._current_arr is not None and
              self._output_dir  is not None and
              not self._loading)
        self._btn_validate.config(state=tk.NORMAL if ok else tk.DISABLED)

    # ── Chargement image ──────────────────────────────────────────────────────

    def _load_current(self):
        p = self._image_list[self._current_idx]
        self._lbl_filename.config(text=p.name)
        self._lbl_counter.config(
            text=f"{self._current_idx + 1} / {len(self._image_list)}")
        self._btn_prev.config(
            state=tk.NORMAL if self._current_idx > 0 else tk.DISABLED)
        self._btn_next.config(
            state=tk.NORMAL if self._current_idx < len(self._image_list) - 1
            else tk.DISABLED)
        self._loading = True
        self._refresh_validate()
        self._lbl_status.config(text="Chargement...", fg=ORANGE)
        threading.Thread(target=self._load_thread, args=(p,), daemon=True).start()

    def _load_thread(self, path: Path):
        arr = np.array(Image.open(path))
        self.after(0, self._on_loaded, arr)

    def _on_loaded(self, arr):
        self._current_arr = arr
        self._loading = False
        self._lbl_status.config(text="")
        self._refresh_validate()
        self._draw()

    # ── Dessin masque ─────────────────────────────────────────────────────────

    def _draw(self):
        if self._current_arr is None:
            return
        mask = apply_threshold(self._current_arr, self._threshold)
        pil  = Image.fromarray(mask, mode="L")
        cw = max(self._canvas.winfo_width(),  1)
        ch = max(self._canvas.winfo_height(), 1)
        pil.thumbnail((cw, ch), Image.NEAREST)
        self._tk_img = ImageTk.PhotoImage(pil)
        cx, cy = cw // 2, ch // 2
        if self._img_id is None:
            self._img_id = self._canvas.create_image(
                cx, cy, anchor="center", image=self._tk_img)
        else:
            self._canvas.itemconfig(self._img_id, image=self._tk_img)
            self._canvas.coords(self._img_id, cx, cy)

    def _on_resize(self, event):
        if self._img_id:
            self._canvas.coords(self._img_id, event.width // 2, event.height // 2)

    # ── Slider ────────────────────────────────────────────────────────────────

    def _on_slider(self, val):
        v = int(float(val))
        self._threshold = v
        self._lbl_val.config(text=str(v))
        if self._debounce_id:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(DEBOUNCE_MS, self._draw)

    # ── Navigation ────────────────────────────────────────────────────────────

    def _prev_image(self):
        if self._current_idx > 0:
            self._current_idx -= 1
            self._load_current()

    def _next_image(self):
        if self._current_idx < len(self._image_list) - 1:
            self._current_idx += 1
            self._load_current()

    # ── Validation / sauvegarde ───────────────────────────────────────────────

    def _validate(self):
        if self._current_arr is None or self._output_dir is None:
            return
        path = self._image_list[self._current_idx]
        self._btn_validate.config(state=tk.DISABLED)
        self._lbl_status.config(text="Traitement...", fg=ORANGE)
        threading.Thread(
            target=self._save_thread,
            args=(self._current_arr.copy(), self._threshold, path),
            daemon=True
        ).start()

    def _save_thread(self, arr, threshold, src_path):
        try:
            mask = apply_threshold(arr, threshold)
            mask = post_process(mask)
            out  = self._output_dir / src_path.name
            Image.fromarray(mask, mode="L").save(out)
            self.after(0, self._on_saved, out)
        except Exception as e:
            self.after(0, self._on_error, str(e))

    def _on_saved(self, out_path):
        self._lbl_status.config(text=f"OK -> {out_path.name}", fg=GREEN)
        self._refresh_validate()
        if self._current_idx < len(self._image_list) - 1:
            self.after(700, self._next_image)

    def _on_error(self, msg):
        self._lbl_status.config(text=f"Erreur: {msg}", fg=RED)
        self._refresh_validate()
        messagebox.showerror("Erreur", msg)


# ─── Lancement ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ThresholdApp()
    app.mainloop()