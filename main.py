import os
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from processor import load_image_gray, save_image_gray, process_image, is_supported_image, SUPPORTED_EXTS

# Optional Pillow for preview
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except Exception:
    HAS_PIL = False


APP_TITLE = "PhotoEditor - выбор лучшего контраста"
POLL_INTERVAL_MS = 3000


def get_desktop_dir():
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(home, "Desktop"),
        os.path.join(home, "OneDrive", "Desktop"),
        os.path.join(home, "OneDrive", "Рабочий стол"),
        os.path.join(home, "Рабочий стол"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return os.path.join(home, "Desktop")


def get_auto_dirs():
    desktop = get_desktop_dir()
    input_dir = os.path.join(desktop, "PhotoEditor")
    output_dir = os.path.join(input_dir, "PhotoEditorReady")
    return input_dir, output_dir


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


class PhotoEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1100x720")
        self.root.minsize(900, 600)

        self.original_img = None
        self.result_img = None
        self.result_name = None
        self.result_score = None
        self.scores = []
        self.current_path = None

        self.auto_running = False
        self.auto_thread = None
        self.auto_stop_event = threading.Event()
        self.processed_signatures = {}  # path -> (mtime, size)
        self._preview_orig_tk = None
        self._preview_res_tk = None
        self._busy = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Button(top, text="Открыть изображение", command=self.open_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Обработать", command=self.process_current).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Сохранить результат", command=self.save_result).pack(side=tk.LEFT, padx=4)

        self.auto_var = tk.BooleanVar(value=False)
        self.auto_btn = ttk.Checkbutton(top, text="Авто-режим (папка на рабочем столе)", variable=self.auto_var, command=self.toggle_auto)
        self.auto_btn.pack(side=tk.LEFT, padx=12)

        ttk.Button(top, text="Открыть папки авто-режима", command=self.open_auto_folders).pack(side=tk.LEFT, padx=4)

        info = ttk.Frame(self.root, padding=(10, 0))
        info.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="Готово. Откройте изображение или включите авто-режим.")
        ttk.Label(info, textvariable=self.status_var, wraplength=1000).pack(anchor=tk.W)

        auto_info = ttk.Frame(self.root, padding=(10, 0))
        auto_info.pack(fill=tk.X)
        in_dir, out_dir = get_auto_dirs()
        self.auto_info_var = tk.StringVar(value=f"Авто-режим: вход: {in_dir} | выход: {out_dir}")
        ttk.Label(auto_info, textvariable=self.auto_info_var, foreground="#444", wraplength=1050).pack(anchor=tk.W)

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(main, padding=5)
        right = ttk.Frame(main, padding=5)
        main.add(left, weight=3)
        main.add(right, weight=2)

        previews = ttk.Frame(left)
        previews.pack(fill=tk.BOTH, expand=True)

        orig_frame = ttk.LabelFrame(previews, text="Исходное изображение")
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.orig_canvas = tk.Label(orig_frame, text="Нет изображения", anchor="center")
        self.orig_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        res_frame = ttk.LabelFrame(previews, text="Лучший результат")
        res_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.res_canvas = tk.Label(res_frame, text="Нет результата", anchor="center")
        self.res_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.result_info_var = tk.StringVar(value="Лучший метод: —")
        ttk.Label(left, textvariable=self.result_info_var, font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(8, 0))

        ttk.Label(right, text="Оценки методов (контраст = std dev):").pack(anchor=tk.W)
        cols = ("method", "score", "status")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=20)
        self.tree.heading("method", text="Метод")
        self.tree.heading("score", text="Контраст")
        self.tree.heading("status", text="Статус")
        self.tree.column("method", width=260)
        self.tree.column("score", width=90, anchor=tk.E)
        self.tree.column("status", width=120)
        self.tree.pack(fill=tk.BOTH, expand=True, pady=(4, 4))

        ttk.Label(right, text="Журнал:").pack(anchor=tk.W)
        self.log_text = tk.Text(right, height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=False)
        self.log_text.configure(state=tk.DISABLED)

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        def _append():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, line)
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
        self.root.after(0, _append)

    def set_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("Все файлы", "*.*")]
        )
        if not path:
            return
        try:
            img = load_image_gray(path)
            self.original_img = img
            self.current_path = path
            self.result_img = None
            self.result_name = None
            self.result_score = None
            self.scores = []
            self._update_previews()
            self._update_scores([])
            self.result_info_var.set("Лучший метод: —")
            self.set_status(f"Загружено: {path}")
            self.log(f"Загружено изображение: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось открыть изображение:\n{e}")

    def _update_previews(self):
        self._show_on_label(self.orig_canvas, self.original_img, is_orig=True)
        self._show_on_label(self.res_canvas, self.result_img, is_orig=False)

    def _show_on_label(self, label, img_gray, is_orig):
        if img_gray is None:
            label.configure(image="", text="Нет изображения" if is_orig else "Нет результата")
            if is_orig:
                self._preview_orig_tk = None
            else:
                self._preview_res_tk = None
            return
        if not HAS_PIL:
            h, w = img_gray.shape[:2]
            label.configure(image="", text=f"{w}x{h}\n(установите Pillow для предпросмотра)")
            return
        # Fit to label size
        lbl_w = max(100, label.winfo_width() - 10)
        lbl_h = max(100, label.winfo_height() - 10)
        h, w = img_gray.shape[:2]
        scale = min(lbl_w / w, lbl_h / h, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img_gray, (nw, nh), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(pil)
        label.configure(image=tk_img, text="")
        if is_orig:
            self._preview_orig_tk = tk_img
        else:
            self._preview_res_tk = tk_img

    def _update_scores(self, scores):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for s in scores:
            score_str = f"{s['score']:.2f}" if s.get('score') is not None else "-"
            status = "OK" if s.get('ok') else "Ошибка"
            self.tree.insert("", tk.END, values=(s['name'], score_str, status))

    def process_current(self):
        if self.original_img is None:
            messagebox.showwarning("Нет изображения", "Сначала откройте изображение.")
            return
        if self._busy:
            return
        self._busy = True
        self.set_status("Обработка... прогоняем все методы.")
        self.log("Запуск обработки всех методов...")

        img_copy = self.original_img.copy()

        def worker():
            try:
                best_img, best_name, best_score, scores = process_image(img_copy)
                def done():
                    self.result_img = best_img
                    self.result_name = best_name
                    self.result_score = best_score
                    self.scores = scores
                    self._update_previews()
                    self._update_scores(scores)
                    self.result_info_var.set(f"Лучший метод: {best_name} | Контраст: {best_score:.2f}")
                    self.set_status(f"Готово. Лучший метод: {best_name} (контраст {best_score:.2f})")
                    self.log(f"Лучший метод: {best_name}, контраст={best_score:.2f}")
                    self._busy = False
                self.root.after(0, done)
            except Exception as e:
                tb = traceback.format_exc()
                def err():
                    self._busy = False
                    self.set_status("Ошибка при обработке.")
                    self.log(f"Ошибка: {e}")
                    messagebox.showerror("Ошибка обработки", f"{e}\n\n{tb}")
                self.root.after(0, err)

        threading.Thread(target=worker, daemon=True).start()

    def save_result(self):
        if self.result_img is None:
            messagebox.showwarning("Нет результата", "Сначала обработайте изображение.")
            return
        initial = "result.png"
        if self.current_path:
            base = os.path.splitext(os.path.basename(self.current_path))[0]
            initial = base + "_best.png"
        path = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".png",
            initialfile=initial,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("BMP", "*.bmp"), ("TIFF", "*.tif *.tiff"), ("WebP", "*.webp")]
        )
        if not path:
            return
        try:
            save_image_gray(path, self.result_img)
            self.set_status(f"Сохранено: {path}")
            self.log(f"Результат сохранён: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{e}")

    # --- Auto mode ---

    def toggle_auto(self):
        if self.auto_var.get():
            self.start_auto()
        else:
            self.stop_auto()

    def start_auto(self):
        in_dir, out_dir = get_auto_dirs()
        try:
            safe_mkdir(in_dir)
            safe_mkdir(out_dir)
        except Exception as e:
            self.auto_var.set(False)
            messagebox.showerror("Ошибка", f"Не удалось создать папки авто-режима:\n{e}")
            return
        self.auto_stop_event.clear()
        self.auto_running = True
        self.log(f"Авто-режим включён. Вход: {in_dir}")
        self.set_status(f"Авто-режим включён. Ожидаю изображения в {in_dir}")
        self.auto_thread = threading.Thread(target=self._auto_loop, daemon=True)
        self.auto_thread.start()
        self._schedule_auto_tick()

    def stop_auto(self):
        self.auto_running = False
        self.auto_stop_event.set()
        self.log("Авто-режим выключен.")
        self.set_status("Авто-режим выключен.")

    def _schedule_auto_tick(self):
        # Keep UI responsive; also refresh auto info
        in_dir, out_dir = get_auto_dirs()
        self.auto_info_var.set(f"Авто-режим: вход: {in_dir} | выход: {out_dir}")
        if self.auto_running:
            self.root.after(POLL_INTERVAL_MS, self._schedule_auto_tick)

    def _auto_loop(self):
        while not self.auto_stop_event.is_set():
            try:
                self._auto_scan_once()
            except Exception as e:
                self.log(f"Ошибка авто-режима: {e}")
            self.auto_stop_event.wait(POLL_INTERVAL_MS / 1000.0)

    def _auto_scan_once(self):
        in_dir, out_dir = get_auto_dirs()
        safe_mkdir(in_dir)
        safe_mkdir(out_dir)
        try:
            entries = os.listdir(in_dir)
        except Exception:
            return
        for name in entries:
            if self.auto_stop_event.is_set():
                return
            src = os.path.join(in_dir, name)
            if os.path.abspath(src) == os.path.abspath(out_dir):
                continue
            if os.path.isdir(src):
                # skip output dir and other dirs
                continue
            if not is_supported_image(src):
                continue
            try:
                st = os.stat(src)
            except Exception:
                continue
            sig = (st.st_mtime_ns if hasattr(st, 'st_mtime_ns') else int(st.st_mtime * 1e9), st.st_size)
            prev = self.processed_signatures.get(os.path.abspath(src))
            if prev == sig:
                continue
            # also skip if output already exists with same base name and newer/equal? We'll process and overwrite.
            self.log(f"[Авто] Обработка: {name}")
            try:
                img = load_image_gray(src)
                best_img, best_name, best_score, _ = process_image(img)
                base = os.path.splitext(name)[0]
                ext = os.path.splitext(name)[1].lower()
                if ext not in SUPPORTED_EXTS:
                    ext = ".png"
                out_name = f"{base}_best{ext}"
                out_path = os.path.join(out_dir, out_name)
                save_image_gray(out_path, best_img)
                self.processed_signatures[os.path.abspath(src)] = sig
                self.log(f"[Авто] Сохранено: {out_name} (метод: {best_name}, контраст: {best_score:.2f})")
                # Update UI with last processed auto result
                def ui_update(im=img, bim=best_img, bn=best_name, bs=best_score, sp=src):
                    self.original_img = im
                    self.result_img = bim
                    self.result_name = bn
                    self.result_score = bs
                    self.current_path = sp
                    self._update_previews()
                    self.result_info_var.set(f"Лучший метод: {bn} | Контраст: {bs:.2f}")
                    self.set_status(f"[Авто] Обработано: {os.path.basename(sp)} -> {bn} ({bs:.2f})")
                self.root.after(0, ui_update)
            except Exception as e:
                self.log(f"[Авто] Ошибка для {name}: {e}")

    def open_auto_folders(self):
        in_dir, out_dir = get_auto_dirs()
        try:
            safe_mkdir(in_dir)
            safe_mkdir(out_dir)
        except Exception:
            pass
        for p in (in_dir, out_dir):
            try:
                if sys.platform.startswith("win"):
                    os.startfile(p)  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    os.system(f'open "{p}"')
                else:
                    os.system(f'xdg-open "{p}"')
            except Exception:
                pass
        self.log(f"Папки авто-режима: {in_dir} ; {out_dir}")

    def on_close(self):
        self.stop_auto()
        self.root.destroy()


def main():
    root = tk.Tk()
    # ttk theme
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    PhotoEditorApp(root)
    # refresh previews on resize
    def on_resize(event):
        # lightweight: only refresh when main window resized
        pass
    root.bind("<Configure>", on_resize)
    root.mainloop()


if __name__ == "__main__":
    main()
