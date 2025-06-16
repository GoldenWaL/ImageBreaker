import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageOps, ImageDraw, ImageTk
import numpy as np
from scipy.spatial import Voronoi
import random
import threading
import os
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import Polygon, box
import sv_ttk


class ModernImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图片碎纸机")
        self.root.geometry("800x600")

        # 设置高DPI支持
        self.root.tk.call('tk', 'scaling', 1.5)
        sv_ttk.set_theme("dark")

        # 主容器
        self.main_frame = ttk.Frame(self.root, padding=(20, 10))
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        self.title_label = ttk.Label(
            self.main_frame,
            text="图片碎纸机",
            font=("Segoe UI", 16, "bold")
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # 模式选择
        self.mode_frame = ttk.LabelFrame(
            self.main_frame,
            text="处理模式",
            padding=(15, 10)
        )
        self.mode_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)

        self.mode_var = tk.StringVar(value="split")

        self.split_radio = ttk.Radiobutton(
            self.mode_frame,
            text="拆分图像",
            variable=self.mode_var,
            value="split",
            command=self.toggle_mode,
            style="Toggle.TButton"
        )
        self.split_radio.pack(side=tk.LEFT, padx=5)

        self.merge_radio = ttk.Radiobutton(
            self.mode_frame,
            text="合并图像",
            variable=self.mode_var,
            value="merge",
            command=self.toggle_mode,
            style="Toggle.TButton"
        )
        self.merge_radio.pack(side=tk.LEFT, padx=5)

        # 拆分参数面板
        self.split_params_frame = ttk.LabelFrame(
            self.main_frame,
            text="拆分参数",
            padding=(15, 10)
        )

        ttk.Label(
            self.split_params_frame,
            text="生成点数:",
            font=("Segoe UI", 9)
        ).grid(row=0, column=0, sticky="w", pady=5)

        self.num_points = ttk.Spinbox(
            self.split_params_frame,
            from_=100,
            to=10000,
            increment=100,
            font=("Segoe UI", 9),
            width=10
        )
        self.num_points.set(2000)
        self.num_points.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(
            self.split_params_frame,
            text="画布数量:",
            font=("Segoe UI", 9)
        ).grid(row=1, column=0, sticky="w", pady=5)

        self.num_canvases = ttk.Spinbox(
            self.split_params_frame,
            from_=1,
            to=20,
            font=("Segoe UI", 9),
            width=10
        )
        self.num_canvases.set(8)
        self.num_canvases.grid(row=1, column=1, padx=5, pady=5)

        # 合并参数面板
        self.merge_params_frame = ttk.LabelFrame(
            self.main_frame,
            text="合并参数",
            padding=(15, 10)
        )

        self.invert_var = tk.BooleanVar()
        ttk.Checkbutton(
            self.merge_params_frame,
            text="颜色反相",
            variable=self.invert_var,
            style="Switch.TCheckbutton"
        ).pack(pady=5)

        # 文件选择
        self.file_frame = ttk.Frame(self.main_frame)

        self.select_btn = ttk.Button(
            self.file_frame,
            text="选择文件",
            command=self.select_files,
            style="Accent.TButton"
        )
        self.select_btn.pack(side=tk.LEFT, padx=5)

        self.file_label = ttk.Label(
            self.file_frame,
            text="未选择文件",
            font=("Segoe UI", 9),
            foreground="#808080"
        )
        self.file_label.pack(side=tk.LEFT, padx=5)

        # 处理按钮
        self.process_btn = ttk.Button(
            self.main_frame,
            text="开始处理",
            command=self.start_processing,
            style="Accent.TButton"
        )

        # 预览区域
        self.preview_frame = ttk.LabelFrame(
            self.main_frame,
            text="预览",
            padding=(15, 10)
        )
        self.preview_frame.grid_columnconfigure(0, weight=1)

        self.preview_label = ttk.Label(
            self.preview_frame,
            text="处理结果将显示在这里",
            font=("Segoe UI", 9),
            foreground="#808080",
            anchor=tk.CENTER
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 布局
        self.toggle_mode()
        self.file_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")
        self.process_btn.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")
        self.preview_frame.grid(row=5, column=0, columnspan=2, pady=10, sticky="nsew")

        # 配置权重
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(5, weight=1)

        # 自定义样式
        self.setup_styles()

    def setup_styles(self):
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Segoe UI", 10), padding=6)
        style.map("Toggle.TButton",
                  background=[("selected", "#0078D4"), ("!selected", "#333333")],
                  foreground=[("selected", "white"), ("!selected", "white")]
                  )
        style.configure("Switch.TCheckbutton", font=("Segoe UI", 9))
        style.configure("TLabelframe", font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("TSpinbox", font=("Segoe UI", 9), padding=5)

    def toggle_mode(self):
        self.split_params_frame.grid_forget()
        self.merge_params_frame.grid_forget()

        if self.mode_var.get() == "split":
            self.split_params_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
            self.file_label.config(text="请选择源图像文件")
        else:
            self.merge_params_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
            self.file_label.config(text="请选择要合并的图像文件")

        self.update_button_states()

    def update_button_states(self):
        if hasattr(self, 'selected_files'):
            if self.mode_var.get() == "split" and len(self.selected_files) == 1:
                self.process_btn.state(["!disabled"])
            elif self.mode_var.get() == "merge" and len(self.selected_files) > 1:
                self.process_btn.state(["!disabled"])
            else:
                self.process_btn.state(["disabled"])
        else:
            self.process_btn.state(["disabled"])

    def select_files(self):
        if self.mode_var.get() == "split":
            file_path = filedialog.askopenfilename(
                title="选择源图像",
                filetypes=[
                    ("图像文件", "*.png;*.jpg;*.jpeg"),
                    ("所有文件", "*.*")
                ]
            )
            if file_path:
                self.selected_files = [file_path]
                self.file_label.config(text=file_path.split('/')[-1])
        else:
            files = filedialog.askopenfilenames(
                title="选择要合并的图像",
                filetypes=[
                    ("图像文件", "*.png;*.jpg;*.jpeg"),
                    ("所有文件", "*.*")
                ]
            )
            if files:
                self.selected_files = files
                self.file_label.config(text=f"已选择 {len(files)} 个文件")

        self.update_button_states()

    def start_processing(self):
        if not hasattr(self, 'selected_files'):
            messagebox.showwarning("警告", "请先选择文件")
            return

        self.process_btn.config(state="disabled", text="处理中...")
        self.root.update()

        if self.mode_var.get() == "split":
            thread = threading.Thread(target=self.process_split)
        else:
            thread = threading.Thread(target=self.process_merge)
        thread.start()

    def process_split(self):
        try:
            num_points = int(self.num_points.get())
            num_canvases = int(self.num_canvases.get())
            cpu_count = os.cpu_count() or 4
            max_workers = min(cpu_count * 2, 16)

            self.root.after(0, lambda: self.show_status("正在拆分图像..."))

            # 添加进度条
            self.progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(
                self.preview_frame,
                variable=self.progress_var,
                maximum=100,
                mode='determinate'
            )
            progress_bar.pack(fill=tk.X, padx=10, pady=5)
            self.root.update()

            def update_progress(progress):
                self.progress_var.set(progress)
                self.root.update()

            process_image_corrected(
                self.selected_files[0],
                num_points=num_points,
                num_canvases=num_canvases,
                max_workers=max_workers,
                progress_callback=update_progress
            )

            progress_bar.pack_forget()
            self.root.after(0, lambda: self.show_preview("canvas_corrected_0.png"))
            self.root.after(0, lambda: messagebox.showinfo("完成", f"已生成 {num_canvases} 张拆分图像"))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
        finally:
            self.root.after(0, lambda: self.process_btn.config(state="normal", text="开始处理"))

    def process_merge(self):
        try:
            cpu_count = os.cpu_count() or 4
            max_workers = min(cpu_count * 2, 16)

            self.root.after(0, lambda: self.show_status("正在合并图像..."))

            # 添加进度条
            self.progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(
                self.preview_frame,
                variable=self.progress_var,
                maximum=100,
                mode='determinate'
            )
            progress_bar.pack(fill=tk.X, padx=10, pady=5)
            self.root.update()

            def update_progress(progress):
                self.progress_var.set(progress)
                self.root.update()

            def load_and_process(file_path):
                img = Image.open(file_path)
                img = make_white_transparent(img)
                if self.invert_var.get():
                    img = invert_image(img)
                return img

            # 分批次处理图像
            batch_size = len(self.selected_files) // max_workers or 1
            file_batches = [self.selected_files[i:i + batch_size]
                            for i in range(0, len(self.selected_files), batch_size)]

            processed_images = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for batch in file_batches:
                    futures.append(executor.submit(
                        lambda files: [load_and_process(f) for f in files], batch
                    ))

                for i, future in enumerate(futures):
                    processed_images.extend(future.result())
                    update_progress((i + 1) / len(futures) * 100)

            base_size = processed_images[0].size
            for img in processed_images[1:]:
                if img.size != base_size:
                    raise ValueError("所有图片尺寸必须相同")

            result = Image.new('RGBA', base_size, (0, 0, 0, 0))
            for img in processed_images:
                result = Image.alpha_composite(result, img)

            result.save('final_composite.png')
            progress_bar.pack_forget()
            self.root.after(0, lambda: self.show_preview('final_composite.png'))
            self.root.after(0, lambda: messagebox.showinfo("完成", "图像合并完成"))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
        finally:
            self.root.after(0, lambda: self.process_btn.config(state="normal", text="开始处理"))

    def show_status(self, message):
        self.preview_label.config(text=message, image=None)
        self.preview_label.image = None

    def show_preview(self, image_path):
        try:
            img = Image.open(image_path)
            img.thumbnail((400, 400))

            mask = Image.new('L', img.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rounded_rectangle([(0, 0), img.size], radius=10, fill=255)

            img.putalpha(mask)
            photo = ImageTk.PhotoImage(img)
            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo

        except Exception as e:
            self.preview_label.config(text=f"无法加载预览: {str(e)}")


def generate_valid_voronoi_polygons(image, num_points):
    """生成在图像边界内的有效 Voronoi 多边形"""
    width, height = image.size
    points = np.random.rand(num_points, 2) * [width, height]
    vor = Voronoi(points)

    boundary = box(0, 0, width, height)
    valid_polygons = []

    for region_index in vor.point_region:
        region = vor.regions[region_index]
        if not region or -1 in region:
            polygon = construct_finite_region(vor, region_index, width, height)
        else:
            polygon = Polygon([vor.vertices[i] for i in region])

        if polygon.is_valid and not polygon.is_empty:
            clipped = polygon.intersection(boundary)
            if clipped.geom_type == 'Polygon':
                valid_polygons.append([(int(x), int(y)) for x, y in clipped.exterior.coords])

    return valid_polygons


def construct_finite_region(vor, region_index, width, height):
    """将包含 -1 的 Voronoi 区域封闭成有限多边形"""
    new_region = []
    center = vor.points[vor.point_region == region_index][0]
    for i in vor.regions[region_index]:
        if i == -1:
            continue
        new_region.append(vor.vertices[i])

    polygon = Polygon(new_region)
    return polygon


def process_image_corrected(image_path, num_points=5000, num_canvases=8, max_workers=8, progress_callback=None):
    try:
        orig_image = Image.open(image_path).convert('RGB')
        inverted = ImageOps.invert(orig_image)
        width, height = orig_image.size
        polygons = generate_valid_voronoi_polygons(orig_image, num_points)
        canvases = [Image.new('RGB', (width, height), 'white') for _ in range(num_canvases)]
        lock = threading.Lock()

        total_polygons = len(polygons)
        processed = 0

        def process_batch(batch):
            nonlocal processed
            batch_results = []
            for polygon in batch:
                mask = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(mask)
                draw.polygon(polygon, fill=255)
                region = Image.new('RGB', (width, height), (255, 255, 255))
                region.paste(inverted, (0, 0), mask=mask)
                temp = Image.new('RGBA', (width, height), (0, 0, 0, 0))
                temp.paste(region, (0, 0), mask=mask)
                idx = random.randint(0, num_canvases - 1)
                batch_results.append((idx, temp))
                processed += 1

                if progress_callback and processed % 100 == 0:
                    progress_callback(processed / total_polygons * 100)

            with lock:
                for idx, temp in batch_results:
                    canvases[idx].paste(
                        Image.alpha_composite(canvases[idx].convert('RGBA'), temp).convert('RGB')
                    )

        batch_size = len(polygons) // (max_workers * 4) or 1
        polygon_batches = [polygons[i:i + batch_size] for i in range(0, len(polygons), batch_size)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(process_batch, polygon_batches)

        for i, canvas in enumerate(canvases):
            #draw = ImageDraw.Draw(canvas)
            #draw.rounded_rectangle([(0, 0), (width - 1, height - 1)], radius=20, outline="#0078D4", width=5)
            canvas.save(f'canvas_corrected_{i}.png')

        if progress_callback:
            progress_callback(100)

    finally:
        orig_image.close()
        inverted.close()


def make_white_transparent(img, threshold=240):
    img = img.convert('RGBA')
    pixels = img.load()
    for i in range(img.width):
        for j in range(img.height):
            r, g, b, a = pixels[i, j]
            if r >= threshold and g >= threshold and b >= threshold:
                pixels[i, j] = (r, g, b, 0)
    return img


def invert_image(img):
    rgb = img.convert('RGB')
    alpha = img.getchannel('A')
    inverted = ImageOps.invert(rgb)
    inverted.putalpha(alpha)
    return inverted


if __name__ == "__main__":
    root = tk.Tk()
    try:
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    app = ModernImageProcessorApp(root)
    root.mainloop()