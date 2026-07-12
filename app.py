"""
app.py
------
Metal Assembly Inspector - Desktop GUI

Simple workflow for the shop floor:
    1. Click "Browse..." to pick Image A (reference / known-good assembly).
    2. Click "Browse..." to pick Image B (assembly being inspected).
    3. Click "Compare".
    4. The side-by-side result appears with every difference boxed and
       highlighted in red on the right-hand image.
    5. Click "Save Result..." to export the composite as a JPEG/PNG.

Can also be run headlessly from the command line for batch/acceptance
testing -- see the bottom of this file / README.md for usage:
    python app.py --cli imageA.jpg imageB.jpg output.jpg
"""

import sys
import os
import argparse
import threading

import cv2
import numpy as np

from compare_core import compare, CONFIG


APP_TITLE = "Metal Assembly Inspector"


# ---------------------------------------------------------------------------
# CLI mode (no display needed - useful for scripted/batch acceptance testing)
# ---------------------------------------------------------------------------
def run_cli(path_a, path_b, out_path):
    img_a = cv2.imread(path_a)
    img_b = cv2.imread(path_b)
    if img_a is None:
        print(f"ERROR: could not read image A: {path_a}")
        sys.exit(1)
    if img_b is None:
        print(f"ERROR: could not read image B: {path_b}")
        sys.exit(1)

    composite, boxes, similarity, confidence = compare(img_a, img_b)
    cv2.imwrite(out_path, composite, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"Alignment confidence : {confidence}")
    print(f"Similarity score     : {similarity:.4f}  (1.0 = identical)")
    print(f"Differences found    : {len(boxes)}")
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        print(f"  #{i}: box at x={x}, y={y}, w={w}, h={h}")
    print(f"Result saved to      : {out_path}")


# ---------------------------------------------------------------------------
# GUI mode
# ---------------------------------------------------------------------------
class InspectorApp:
    """Lazily imports tkinter/Pillow-Tk bindings so this module can still be
    used in --cli mode on machines/environments without a display or Tk
    installed (e.g. a headless CI box running acceptance tests)."""

    def __new__(cls):
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        from PIL import Image, ImageTk

        class _InspectorApp(tk.Tk):
            def __init__(self):
                super().__init__()
                self.title(APP_TITLE)
                self.geometry("1150x780")
                self.minsize(820, 560)
                self.configure(bg="#1e1e1e")

                self.path_a = None
                self.path_b = None
                self.result_bgr = None      # last composite (OpenCV BGR array)
                self.last_stats = None

                self._build_ui()

            # -- UI construction ---------------------------------------------------
            def _build_ui(self):
                style = ttk.Style(self)
                try:
                    style.theme_use("clam")
                except tk.TclError:
                    pass
                style.configure("TButton", padding=6)
                style.configure("Status.TLabel", background="#1e1e1e", foreground="#dddddd")

                top = tk.Frame(self, bg="#1e1e1e")
                top.pack(side="top", fill="x", padx=10, pady=8)

                self.btn_a = ttk.Button(top, text="Browse Image A (reference)...",
                                         command=self.pick_image_a)
                self.btn_a.pack(side="left", padx=(0, 8))
                self.lbl_a = ttk.Label(top, text="No file selected", style="Status.TLabel")
                self.lbl_a.pack(side="left", padx=(0, 24))

                self.btn_b = ttk.Button(top, text="Browse Image B (inspected)...",
                                         command=self.pick_image_b)
                self.btn_b.pack(side="left", padx=(0, 8))
                self.lbl_b = ttk.Label(top, text="No file selected", style="Status.TLabel")
                self.lbl_b.pack(side="left", padx=(0, 24))

                self.btn_compare = ttk.Button(top, text="Compare", command=self.run_compare)
                self.btn_compare.pack(side="left", padx=(0, 8))

                self.btn_save = ttk.Button(top, text="Save Result...",
                                            command=self.save_result, state="disabled")
                self.btn_save.pack(side="left")

                # image display area
                self.canvas = tk.Canvas(self, bg="#111111", highlightthickness=0)
                self.canvas.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 6))
                self._tk_img = None  # keep reference to avoid GC

                # status bar
                self.status = ttk.Label(self, text="Select two images, then click Compare.",
                                         style="Status.TLabel", anchor="w")
                self.status.pack(side="bottom", fill="x", padx=12, pady=(0, 8))

                self.canvas.bind("<Configure>", lambda e: self._redraw())

            # -- file pickers --------------------------------------------------------
            def pick_image_a(self):
                path = filedialog.askopenfilename(
                    title="Select reference Image A",
                    filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
                if path:
                    self.path_a = path
                    self.lbl_a.config(text=os.path.basename(path))

            def pick_image_b(self):
                path = filedialog.askopenfilename(
                    title="Select inspected Image B",
                    filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
                if path:
                    self.path_b = path
                    self.lbl_b.config(text=os.path.basename(path))

            # -- compare -------------------------------------------------------------
            def run_compare(self):
                if not self.path_a or not self.path_b:
                    messagebox.showwarning(APP_TITLE, "Please select both Image A and Image B first.")
                    return

                self.btn_compare.config(state="disabled")
                self.status.config(text="Comparing... this usually takes a couple of seconds.")
                self.update_idletasks()

                # run on a background thread so the UI doesn't freeze
                threading.Thread(target=self._compare_worker, daemon=True).start()

            def _compare_worker(self):
                try:
                    img_a = cv2.imread(self.path_a)
                    img_b = cv2.imread(self.path_b)
                    if img_a is None or img_b is None:
                        raise ValueError("One of the selected files could not be read as an image.")

                    composite, boxes, similarity, confidence = compare(img_a, img_b)
                    self.result_bgr = composite
                    self.last_stats = (boxes, similarity, confidence)
                    self.after(0, self._on_compare_done, None)
                except Exception as e:
                    self.after(0, self._on_compare_done, e)

            def _on_compare_done(self, error):
                self.btn_compare.config(state="normal")
                if error is not None:
                    messagebox.showerror(APP_TITLE, f"Comparison failed:\n{error}")
                    self.status.config(text="Comparison failed. See error dialog.")
                    return

                boxes, similarity, confidence = self.last_stats
                conf_text = {"ok": "good", "low": "caution - viewpoints differ",
                             "none": "poor - could not align images"}[confidence]
                self.status.config(
                    text=(f"Found {len(boxes)} difference region(s)  |  "
                          f"similarity score: {similarity:.3f}  |  "
                          f"alignment confidence: {conf_text}"))
                self.btn_save.config(state="normal")
                self._redraw()

            # -- display / save --------------------------------------------------------
            def _redraw(self):
                if self.result_bgr is None:
                    return
                cw = max(self.canvas.winfo_width(), 100)
                ch = max(self.canvas.winfo_height(), 100)

                rgb = cv2.cvtColor(self.result_bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                scale = min(cw / img.width, ch / img.height)
                scale = min(scale, 1.0) if scale > 0 else 1.0
                new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
                img = img.resize(new_size, Image.LANCZOS)

                self._tk_img = ImageTk.PhotoImage(img)
                self.canvas.delete("all")
                self.canvas.create_image(cw // 2, ch // 2, image=self._tk_img, anchor="center")

            def save_result(self):
                if self.result_bgr is None:
                    return
                path = filedialog.asksaveasfilename(
                    title="Save comparison result",
                    defaultextension=".jpg",
                    filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
                if not path:
                    return
                ok = cv2.imwrite(path, self.result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    messagebox.showinfo(APP_TITLE, f"Saved to:\n{path}")
                else:
                    messagebox.showerror(APP_TITLE, "Failed to save file.")

        return _InspectorApp()


def main():
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--cli", action="store_true",
                         help="Run headlessly without a GUI window.")
    parser.add_argument("image_a", nargs="?", help="(--cli mode) path to reference image")
    parser.add_argument("image_b", nargs="?", help="(--cli mode) path to inspected image")
    parser.add_argument("output", nargs="?", help="(--cli mode) path to save result image")
    args = parser.parse_args()

    if args.cli:
        if not (args.image_a and args.image_b and args.output):
            parser.error("--cli requires: image_a image_b output")
        run_cli(args.image_a, args.image_b, args.output)
    else:
        app = InspectorApp()
        app.mainloop()


if __name__ == "__main__":
    main()
