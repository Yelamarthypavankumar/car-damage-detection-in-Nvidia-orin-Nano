import sys
sys.path.append("../Training")
import streamlit as st
from PIL import Image
import numpy as np
import os
import cv2
import time
import tempfile
from collections import Counter
import pycuda.driver as cuda
import tensorrt as trt
from yolox.data.data_augment import preproc as preprocess
from yolox.utils import multiclass_nms, vis

# ---------------- Config ----------------
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
IMAGE_DIR = './examples'
ENGINE_PATH = "car_trt.trt"
COCO_CLASSES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
INPUT_SHAPE = (640, 640)
SCORE_THR = 0.3
NMS_THR = 0.45


# ---------------- CUDA Context Manager ----------------
class CudaContextManager:
    def __enter__(self):
        cuda.init()
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()
        return self.ctx

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ctx.pop()
        del self.ctx

# ---------------- TRT Inference Utils ----------------
def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size = trt.volume(shape)
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append({'name': name, 'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'name': name, 'host': host_mem, 'device': device_mem})

    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    for inp in inputs:
        cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
        context.set_tensor_address(inp['name'], int(inp['device']))
    for out in outputs:
        context.set_tensor_address(out['name'], int(out['device']))
    context.execute_async_v3(stream.handle)
    for out in outputs:
        cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
    stream.synchronize()
    return [out['host'] for out in outputs]

# ---------------- Postprocess (8400 anchors) ----------------
def demo_postprocess_8400(outputs, img_size):
    grids = []
    expanded_strides = []
    strides = [8, 16, 32]
    hs_ws = [(80, 80), (40, 40), (20, 20)]

    for (hsize, wsize), stride in zip(hs_ws, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, axis=1)
    expanded_strides = np.concatenate(expanded_strides, axis=1)

    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs

# ---------------- Streamlit App ----------------
st.set_page_config(layout="wide")

if 'selected_media' not in st.session_state:
    st.session_state.selected_media = None

if 'media_type' not in st.session_state:
    st.session_state.media_type = 'Image'

col_logo, col_media_type = st.columns([1, 3])


with col_media_type:
    st.title("Car Damage Detection ")
    media_type = "Image"

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Upload or Select a File")
    uploaded_file = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
        st.session_state.selected_media = temp_file.name

    for filename in os.listdir(IMAGE_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            media_path = os.path.join(IMAGE_DIR, filename)
            img = Image.open(media_path)
            img.thumbnail((80, 80))
            st.image(img, width=80)
            if st.button(f"{filename}", key=f"select_{filename}"):
                st.session_state.selected_media = media_path

with col2:
    st.subheader("Selected Image")
    if st.session_state.selected_media:
        st.image(st.session_state.selected_media, caption="Selected", use_container_width=True)
    process_button = st.button("Submit")

with col3:
    st.subheader("Processed Image")
    if process_button and st.session_state.selected_media:
        try:
            with CudaContextManager():
                with st.spinner("Running inference..."):
                    engine = load_engine(ENGINE_PATH)
                    context = engine.create_execution_context()
                    inputs, outputs, bindings, stream = allocate_buffers(engine)

                    origin_img = cv2.imread(st.session_state.selected_media)
                    img, ratio = preprocess(origin_img, INPUT_SHAPE)
                    img = img.astype(np.float32)
                    inputs[0]['host'][:] = img.ravel()

                    # --------- Start inference timing ---------
                    start_time = time.perf_counter()

                    trt_outputs = do_inference(context, bindings, inputs, outputs, stream)

                    end_time = time.perf_counter()
                    inference_time_ms = (end_time - start_time) * 1000
                    # --------- End inference timing ---------

                    output = trt_outputs[0].reshape(1, -1, 5 + len(COCO_CLASSES))
                    predictions = demo_postprocess_8400(output, INPUT_SHAPE)[0]

                    boxes = predictions[:, :4]
                    scores = predictions[:, 4:5] * predictions[:, 5:]

                    boxes_xyxy = np.ones_like(boxes)
                    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
                    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
                    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
                    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
                    boxes_xyxy /= ratio

                    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=NMS_THR, score_thr=SCORE_THR)
                    origin_img_vis = origin_img.copy()

                    if dets is not None:
                        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                        origin_img_vis = vis(origin_img_vis, final_boxes, final_scores, final_cls_inds, conf=SCORE_THR, class_names=COCO_CLASSES)
                        cls_counts = Counter([COCO_CLASSES[int(c)] for c in final_cls_inds])

                        st.image(cv2.cvtColor(origin_img_vis, cv2.COLOR_BGR2RGB), caption="Detection Result", use_container_width=True)
                        st.markdown("### 🧾 Defect Summary")
                        for cls, count in cls_counts.items():
                            st.markdown(f"- **{cls}**: {count}")
                    else:
                        st.success("✅ No defects detected.")

                    # Show inference time
                    st.markdown(f"🕒 **Inference Time:** `{inference_time_ms:.2f} ms`")

        except Exception as e:
            import traceback
            st.error(f"Inference failed: {e}")
            st.text(traceback.format_exc())

