import streamlit as st
from skimage import io
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt  # 新增這一行
from ultralytics.utils.plotting import Annotator
from PIL import Image #1024
#新增20260310
import os
import sys

# 自動安裝缺失的系統庫 (針對 Streamlit Cloud 環境)
def install_packages():
    try:
        # 嘗試檢查是否有 libGL，若無則嘗試用 pip 補救或提醒
        import cv2
    except ImportError:
        # 強制重新安裝 headless 版本確保路徑正確
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless", "--force-reinstall"])

# 執行安裝
install_packages()
# 強制告訴環境我們不需要 GUI
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# 【核心修正】告訴 PyTorch 許可 Ultralytics 的自定義類別
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception:
    pass
# 2. 如果系統找不到 libgthread，嘗試動態加載一個通用的 glib
# 這是一個小撇步，有時能讓 Python 內部機制跳過檢查
try:
    import ctypes
    ctypes.CDLL('libgthread-2.0.so.0', mode=os.RTLD_GLOBAL)
except Exception:
    pass
import streamlit as st
st.title("上傳圖片辨識")
st.info("YOLO_V8)， 80 個類別的物體檢測，這些類別包括人，動物，交通工具，家具等")

uploaded_file = st.file_uploader("上傳圖片", type=['jpg', 'jpeg', 'png', 'gif'])
colors_rgb = {str(i): ((i * 30) % 256, (i * 50) % 256, (i * 70) % 256) for i in range(30)}

if uploaded_file is not None:
    image = io.imread(uploaded_file)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    
    # 【關鍵修正 1】確保記憶體連續，否則 Annotator 會報錯
    image = np.ascontiguousarray(image)

    # 載入模型
    pretrained_weights_path = './model/yolov8n.pt'
    yolo = YOLO(pretrained_weights_path) # 建議直接載入，更簡潔
    
    try:
        results = yolo.predict(image)
        st.subheader("檢測結果")
        
        # 【關鍵修正 2】Annotator 應該放在迴圈外面，不然每一框都會蓋掉前一框
        annotator = Annotator(image)
        
        result_str = []
        colors_count = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                xyxy = box.xyxy[0]
                label = yolo.names[cls]
                result_str.append(label)
                
                # 繪製標籤
                annotator.box_label(xyxy, label, colors_rgb.get(str(colors_count), (255, 0, 0)))
                colors_count += 1
        
        # 取得標註後的圖片 (BGR 轉 RGB)
        img_result = annotator.result()
        
        # 顯示圖片
        st.image(img_result, caption="檢測結果", use_container_width=True)
        st.write(f"偵測到物品: {', '.join(result_str)}")
        
    except Exception as e:
        st.error(f"預測過程中發生錯誤: {e}")



