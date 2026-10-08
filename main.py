import cv2
import numpy as np
from skimage import img_as_float
from skimage.filters import frangi
import streamlit as st

# ===================== 页面初始化 =====================
st.set_page_config(layout="wide", page_title="眼底图像处理系统")
st.title("眼底图像处理系统")

# 会话状态：保存图像数据，对应原全局变量
if "origin_image" not in st.session_state:
    st.session_state.origin_image = None
if "gray_image" not in st.session_state:
    st.session_state.gray_image = None
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_title" not in st.session_state:
    st.session_state.current_title = "当前图像：无"


# ===================== 图像处理函数（算法完全复用原逻辑） =====================
def update_display(img, title):
    """更新当前显示的图像和标题"""
    st.session_state.current_image = img
    st.session_state.current_title = f"当前图像：{title}"


def load_image(uploaded_file):
    """读取上传的图像"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("图片读取失败！")
        return
    st.session_state.origin_image = image
    st.session_state.gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    update_display(image, "原始眼底图像")


def to_gray():
    if st.session_state.origin_image is None:
        st.info("请先上传原图")
        return
    gray = st.session_state.gray_image
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    update_display(gray_bgr, "灰度图像")


def histogram_equalization():
    if st.session_state.gray_image is None:
        st.info("请先转为灰度图")
        return
    equalized = cv2.equalizeHist(st.session_state.gray_image)
    equalized_bgr = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    update_display(equalized_bgr, "直方图均衡图像")


def brighten():
    if st.session_state.gray_image is None:
        st.info("请先转为灰度图")
        return
    brightened = cv2.add(st.session_state.gray_image, 50)
    brightened_bgr = cv2.cvtColor(brightened, cv2.COLOR_GRAY2BGR)
    update_display(brightened_bgr, "加亮图像")


def darken():
    if st.session_state.gray_image is None:
        st.info("请先转为灰度图")
        return
    darkened = cv2.subtract(st.session_state.gray_image, 50)
    darkened_bgr = cv2.cvtColor(darkened, cv2.COLOR_GRAY2BGR)
    update_display(darkened_bgr, "变暗图像")


def contrast_up():
    if st.session_state.gray_image is None:
        st.info("请先转为灰度图")
        return
    multiplied = cv2.multiply(st.session_state.gray_image, 1.5)
    multiplied = np.clip(multiplied, 0, 255).astype(np.uint8)
    multiplied_bgr = cv2.cvtColor(multiplied, cv2.COLOR_GRAY2BGR)
    update_display(multiplied_bgr, "对比增强图像")


def contrast_down():
    if st.session_state.gray_image is None:
        st.info("请先转为灰度图")
        return
    divided = cv2.divide(st.session_state.gray_image, 2)
    divided_bgr = cv2.cvtColor(divided, cv2.COLOR_GRAY2BGR)
    update_display(divided_bgr, "对比降低图像")


def logic_ops(op):
    if st.session_state.gray_image is None:
        st.info("请先转为灰度图")
        return
    _, binary = cv2.threshold(st.session_state.gray_image, 127, 255, cv2.THRESH_BINARY)
    if op == 'not':
        result = cv2.bitwise_not(binary)
        title = "逻辑非图像"
    elif op == 'and':
        result = cv2.bitwise_and(st.session_state.gray_image, binary)
        title = "逻辑与图像"
    elif op == 'or':
        result = cv2.bitwise_or(st.session_state.gray_image, binary)
        title = "逻辑或图像"
    else:
        result = binary
        title = "二值图像"
    result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    update_display(result_bgr, title)


def pseudo_color():
    if st.session_state.gray_image is None:
        st.info("请先转为灰度图")
        return
    color_map = cv2.applyColorMap(st.session_state.gray_image, cv2.COLORMAP_JET)
    update_display(color_map, "伪彩色图像")


def segment_vessels():
    if st.session_state.origin_image is None:
        st.info("请先上传原图")
        return
    green_channel = st.session_state.origin_image[:, :, 1]
    blurred = cv2.GaussianBlur(green_channel, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    vessel_bgr = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    update_display(vessel_bgr, "粗分割血管图像")


def segment_vessels_frangi():
    if st.session_state.origin_image is None:
        st.info("请先上传原图")
        return
    green = st.session_state.origin_image[:, :, 1]
    green_float = img_as_float(green)
    vessels = frangi(green_float)
    vessels_normalized = (vessels - vessels.min()) / (vessels.max() - vessels.min())
    vessels_gray = np.uint8(vessels_normalized * 255)
    equalized = cv2.equalizeHist(vessels_gray)
    update_display(cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR), "精细分割血管图像")


# ===================== 界面布局 =====================
# 左侧侧边栏：所有操作按钮（对应原右侧按钮区）
with st.sidebar:
    st.header("操作面板")

    # 基础操作
    st.subheader("基础操作")
    uploaded = st.file_uploader("上传图像", type=["png", "jpg", "jpeg", "tif", "bmp"])
    if uploaded:
        load_image(uploaded)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("转灰度图", use_container_width=True):
            to_gray()
    with col2:
        if st.session_state.current_image is not None:
            # 下载按钮
            success, encoded_img = cv2.imencode('.png', st.session_state.current_image)
            st.download_button(
                label="保存图像",
                data=encoded_img.tobytes(),
                file_name="处理结果.png",
                mime="image/png",
                use_container_width=True
            )

    # 图像增强
    st.subheader("图像增强")
    if st.button("直方图均衡", use_container_width=True):
        histogram_equalization()
    col3, col4 = st.columns(2)
    with col3:
        if st.button("加亮", use_container_width=True):
            brighten()
    with col4:
        if st.button("变暗", use_container_width=True):
            darken()
    col5, col6 = st.columns(2)
    with col5:
        if st.button("对比增强", use_container_width=True):
            contrast_up()
    with col6:
        if st.button("对比降低", use_container_width=True):
            contrast_down()

    # 逻辑运算
    st.subheader("逻辑运算")
    col7, col8, col9 = st.columns(3)
    with col7:
        if st.button("非", use_container_width=True):
            logic_ops('not')
    with col8:
        if st.button("与", use_container_width=True):
            logic_ops('and')
    with col9:
        if st.button("或", use_container_width=True):
            logic_ops('or')
    if st.button("二值化", use_container_width=True):
        logic_ops('binary')

    # 特殊处理
    st.subheader("特殊处理")
    if st.button("伪彩色处理", use_container_width=True):
        pseudo_color()
    if st.button("血管分割(粗)", use_container_width=True):
        segment_vessels()
    if st.button("血管分割(细)", use_container_width=True):
        segment_vessels_frangi()

# 主区域：图像显示（对应原左侧显示区）
st.subheader(st.session_state.current_title)
if st.session_state.current_image is not None:
    # BGR转RGB用于网页显示
    show_img = cv2.cvtColor(st.session_state.current_image, cv2.COLOR_BGR2RGB)
    st.image(show_img, use_column_width=True)
else:
    st.info("请在左侧上传眼底图像开始处理")
