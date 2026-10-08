import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from skimage import img_as_float
from skimage.filters import frangi

# ===================== 全局配色与样式 =====================
BG_COLOR = "#e8f0fa"    # 窗口背景色
BTN_BG = "#3478f5"      # 按钮背景色
BTN_FG = "white"        # 按钮文字色
FONT_STYLE = ("微软雅黑", 9)
TEXT_FONT = ("微软雅黑", 11, "bold")

# 全局变量
current_show_img = None
image = None
gray = None

# ===================== 通用函数 =====================
# 创建统一样式按钮
def create_btn(parent, txt, cmd):
    return tk.Button(
        parent, text=txt, command=cmd,
        bg=BTN_BG, fg=BTN_FG,
        font=FONT_STYLE, width=12,
        relief=tk.RAISED, bd=2
    )

# 更新图像标题文字
def set_img_title(text):
    img_title_label.config(text=f"当前图像：{text}")

# 图像显示函数
def show_image(cv_img):
    global current_show_img
    current_show_img = cv_img.copy()

    max_width = 420
    height, width = cv_img.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        cv_img = cv2.resize(cv_img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    b, g, r = cv2.split(cv_img)
    img = cv2.merge((r, g, b))
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)

    panel.config(image=imgtk)
    panel.image = imgtk

def show_image_in_window(title, img):
    max_width = 350
    height, width = img.shape[:2]
    if width > max_width:
        scale = max_width / width
        img = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

    if len(img.shape) == 2:
        im = Image.fromarray(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)

    imgtk = ImageTk.PhotoImage(image=im)
    win = tk.Toplevel(root)
    win.title(title)
    label = tk.Label(win, image=imgtk)
    label.image = imgtk
    label.pack()

# ===================== 功能逻辑函数 =====================
# 打开图像
def load_image():
    global image, gray
    file_path = filedialog.askopenfilename(
        filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.tif;*.bmp")]
    )
    if not file_path:
        return
    image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        messagebox.showerror("错误", "图片读取失败！")
        return
    show_image(image)
    set_img_title("原始眼底图像")

# 保存图像
def save_image():
    global current_show_img
    if current_show_img is None:
        messagebox.showinfo("提示", "当前没有可保存的图像，请先打开/处理图片！")
        return
    save_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[
            ("PNG 图片", "*.png"),
            ("JPG 图片", "*.jpg;*.jpeg"),
            ("TIF 图片", "*.tif"),
            ("BMP 图片", "*.bmp")
        ]
    )
    if not save_path:
        return
    cv2.imencode('.png', current_show_img)[1].tofile(save_path)
    messagebox.showinfo("成功", "图像保存完成！")

# 转灰度图
def to_gray():
    global gray
    if image is None:
        messagebox.showinfo("提示", "请先打开原图")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    show_image(gray_bgr)
    set_img_title("灰度图像")

# 直方图均衡
def histogram_equalization():
    if gray is None:
        messagebox.showinfo("提示", "请先转为灰度图")
        return
    equalized = cv2.equalizeHist(gray)
    equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    show_image(equalized_bgr)
    set_img_title("直方图均衡图像")

# 加亮
def brighten():
    if gray is None:
        messagebox.showinfo("提示", "请先转为灰度图")
        return
    brightened = cv2.add(gray, 50)
    brightened_bgr = cv2.cvtColor(brightened, cv2.COLOR_GRAY2BGR)
    show_image(brightened_bgr)
    set_img_title("加亮图像")

# 变暗
def darken():
    if gray is None:
        messagebox.showinfo("提示", "请先转为灰度图")
        return
    darkened = cv2.subtract(gray, 50)
    darkened_bgr = cv2.cvtColor(darkened, cv2.COLOR_GRAY2BGR)
    show_image(darkened_bgr)
    set_img_title("变暗图像")

# 对比增强
def contrast_up():
    if gray is None:
        messagebox.showinfo("提示", "请先转为灰度图")
        return
    multiplied = cv2.multiply(gray, 1.5)
    multiplied = np.clip(multiplied, 0, 255).astype(np.uint8)
    multiplied_bgr = cv2.cvtColor(multiplied, cv2.COLOR_GRAY2BGR)
    show_image(multiplied_bgr)
    set_img_title("对比增强图像")

# 对比降低
def contrast_down():
    if gray is None:
        messagebox.showinfo("提示", "请先转为灰度图")
        return
    divided = cv2.divide(gray, 2)
    divided_bgr = cv2.cvtColor(divided, cv2.COLOR_GRAY2BGR)
    show_image(divided_bgr)
    set_img_title("对比降低图像")

# 逻辑运算
def logic_ops(op):
    if gray is None:
        messagebox.showinfo("提示", "请先转为灰度图")
        return
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    if op == 'not':
        result = cv2.bitwise_not(binary)
        title = "逻辑非图像"
    elif op == 'and':
        result = cv2.bitwise_and(gray, binary)
        title = "逻辑与图像"
    elif op == 'or':
        result = cv2.bitwise_or(gray, binary)
        title = "逻辑或图像"
    else:
        result = binary
        title = "二值图像"
    result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    show_image(result_bgr)
    set_img_title(title)

# 伪彩色处理
def pseudo_color():
    if gray is None:
        messagebox.showinfo("提示", "请先转为灰度图")
        return
    color_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    show_image(color_map)
    set_img_title("伪彩色图像")

# 粗血管分割
def segment_vessels():
    if image is None:
        messagebox.showinfo("提示", "请先打开原图")
        return
    green_channel = image[:, :, 1]
    blurred = cv2.GaussianBlur(green_channel, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    vessel_bgr = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    show_image(vessel_bgr)
    set_img_title("粗分割血管图像")

# 细血管分割
def segment_vessels_frangi():
    global image
    if image is None:
        messagebox.showinfo("提示", "请先打开原图")
        return
    green = image[:, :, 1]
    green_float = img_as_float(green)
    vessels = frangi(green_float)
    vessels_normalized = (vessels - vessels.min()) / (vessels.max() - vessels.min())
    vessels_gray = np.uint8(vessels_normalized * 255)
    equalized = cv2.equalizeHist(vessels_gray)
    show_image(cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR))
    set_img_title("精细分割血管图像")

# ===================== 主界面布局（左图 + 右按钮） =====================
root = tk.Tk()
root.title("眼底图像处理GUI系统")
root.geometry("1100x720")
root.configure(bg=BG_COLOR)

# 整体左右分栏容器
main_frame = tk.Frame(root, bg=BG_COLOR)
main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

# -------- 左侧区域：图像展示区 --------
left_frame = tk.Frame(main_frame, bg=BG_COLOR)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# 图像文字标注
img_title_label = tk.Label(left_frame, text="当前图像：无", bg=BG_COLOR, font=TEXT_FONT)
img_title_label.pack(pady=10)

# 图片显示面板
panel = tk.Label(left_frame, bg="white", relief=tk.GROOVE, bd=2)
panel.pack(pady=5)

# -------- 右侧区域：所有按钮 纵向整齐排列 --------
right_frame = tk.Frame(main_frame, bg=BG_COLOR)
right_frame.pack(side=tk.RIGHT, padx=(20, 0), fill=tk.Y)

# 分组1：基础操作
group1_label = tk.Label(right_frame, text="基础操作", bg=BG_COLOR, font=("微软雅黑", 10, "bold"))
group1_label.pack(pady=(0, 8))
create_btn(right_frame, "打开图像", load_image).pack(fill=tk.X, pady=3)
create_btn(right_frame, "保存图像", save_image).pack(fill=tk.X, pady=3)
create_btn(right_frame, "转灰度图", to_gray).pack(fill=tk.X, pady=3)

# 分组2：图像增强
group2_label = tk.Label(right_frame, text="图像增强", bg=BG_COLOR, font=("微软雅黑", 10, "bold"))
group2_label.pack(pady=(12, 8))
create_btn(right_frame, "直方图均衡", histogram_equalization).pack(fill=tk.X, pady=3)
create_btn(right_frame, "加亮", brighten).pack(fill=tk.X, pady=3)
create_btn(right_frame, "变暗", darken).pack(fill=tk.X, pady=3)
create_btn(right_frame, "对比增强", contrast_up).pack(fill=tk.X, pady=3)
create_btn(right_frame, "对比降低", contrast_down).pack(fill=tk.X, pady=3)

# 分组3：逻辑运算
group3_label = tk.Label(right_frame, text="逻辑运算", bg=BG_COLOR, font=("微软雅黑", 10, "bold"))
group3_label.pack(pady=(12, 8))
create_btn(right_frame, "逻辑非", lambda: logic_ops('not')).pack(fill=tk.X, pady=3)
create_btn(right_frame, "逻辑与", lambda: logic_ops('and')).pack(fill=tk.X, pady=3)
create_btn(right_frame, "逻辑或", lambda: logic_ops('or')).pack(fill=tk.X, pady=3)

# 分组4：特殊处理
group4_label = tk.Label(right_frame, text="特殊处理", bg=BG_COLOR, font=("微软雅黑", 10, "bold"))
group4_label.pack(pady=(12, 8))
create_btn(right_frame, "伪彩色处理", pseudo_color).pack(fill=tk.X, pady=3)
create_btn(right_frame, "血管分割(粗)", segment_vessels).pack(fill=tk.X, pady=3)
create_btn(right_frame, "血管分割(细)", segment_vessels_frangi).pack(fill=tk.X, pady=3)

root.mainloop()
