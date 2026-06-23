import tensorflow as tf
import sys
import os
import json
import cv2
import torch
import math
import shutil
import numpy as np
import gc
from PIL import Image

# --- BƯỚC LỌC AN TOÀN CHO FLORENCE-2 ---
import transformers
import transformers.dynamic_module_utils as dynamic_utils
from transformers import AutoProcessor, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer, util

orig_get_imports = dynamic_utils.get_imports
def custom_get_imports(filename):
    imports = orig_get_imports(filename)
    if imports is not None and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports
dynamic_utils.get_imports = custom_get_imports

# --- CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU ---
BASE_WORKSPACE = r"G:\.shortcut-targets-by-id\11I5_AMfAufb6crT2hzGrLEI3tMsTsKjX\AIC2026"

DRIVE_INPUT_FOLDER = os.path.join(BASE_WORKSPACE, "OldData\L28")
DRIVE_OUTPUT_FOLDER = os.path.join(BASE_WORKSPACE, "metadata")
DRIVE_KEYFRAMES_META_FOLDER = os.path.join(BASE_WORKSPACE, "keyframes_meta")
LOCAL_TEMP_FOLDER = "temp_processing_videos"

os.makedirs(DRIVE_INPUT_FOLDER, exist_ok=True)
os.makedirs(DRIVE_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DRIVE_KEYFRAMES_META_FOLDER, exist_ok=True)
os.makedirs(LOCAL_TEMP_FOLDER, exist_ok=True)

# Thêm TransNetV2 vào đường dẫn
sys.path.append(os.path.join(os.getcwd(), 'TransNetV2', 'inference'))
from transnetv2 import TransNetV2

# --- KHỞI TẠO MÔ HÌNH ---
print("🚀 Đang khởi tạo và kiểm tra phần cứng cho các mô hình AI...")
if torch.cuda.is_available():
    device = "cuda:0"
    torch_dtype = torch.float16
    print(f"  ✅ [PyTorch] Đã nhận diện GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    torch_dtype = torch.float32
    print("  ⚠️ [PyTorch] KHÔNG tìm thấy GPU!")

tf_gpus = tf.config.list_physical_devices('GPU')
if tf_gpus:
    print(f"  ✅ [TensorFlow] Đã nhận diện được GPU.")
else:
    print("  ⚠️ [TensorFlow] KHÔNG tìm thấy GPU! TransNetV2 sẽ chạy trên CPU.")
print("-" * 50)

print("1/3. Đang tải TransNetV2...")
transnet = TransNetV2()

florence_model_id = "microsoft/Florence-2-base-ft"
print(f"2/3. Đang tải {florence_model_id} trên {device}...")
florence_model = AutoModelForCausalLM.from_pretrained(florence_model_id, torch_dtype=torch_dtype, trust_remote_code=True).to(device).eval()
florence_processor = AutoProcessor.from_pretrained(florence_model_id, trust_remote_code=True)

print("3/3. Đang tải all-MiniLM-L6-v2...")
embedder = SentenceTransformer('all-MiniLM-L6-v2').to(device)
print("✅ TẢI MÔ HÌNH THÀNH CÔNG!\n")


# --- CÁC HÀM XỬ LÝ CỐT LÕI ---
def extract_keyframes_from_shots(video_path, scenes):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    keyframes = []
    
    for shot_idx, (start_frame, end_frame) in enumerate(scenes):
        mid_frame = (start_frame + end_frame) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            keyframes.append({
                "shot_id": f"shot_{shot_idx+1:04d}",
                "start_time": round(start_frame / fps, 2),
                "end_time": round(end_frame / fps, 2),
                "image": pil_img
            })
    cap.release()
    return keyframes, fps

def generate_caption(image):
    prompt = "<MORE_DETAILED_CAPTION>"
    inputs = florence_processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=1024, do_sample=False, num_beams=3
        )
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return florence_processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))[prompt]

def group_shots_to_segments(shots, threshold=0.75):
    if not shots: return []
    segments = []
    current_segment = {
        "segment_id": "seg_0001",
        "start_time": shots[0]["start_time"],
        "end_time": shots[0]["end_time"],
        "shots": [shots[0]]
    }
    
    seg_counter = 1
    for i in range(1, len(shots)):
        embeddings = embedder.encode([shots[i-1]["caption"], shots[i]["caption"]], convert_to_tensor=True)
        if util.cos_sim(embeddings[0], embeddings[1]).item() >= threshold:
            current_segment["shots"].append(shots[i])
            current_segment["end_time"] = shots[i]["end_time"]
        else:
            segments.append(current_segment)
            seg_counter += 1
            current_segment = {
                "segment_id": f"seg_{seg_counter:04d}",
                "start_time": shots[i]["start_time"],
                "end_time": shots[i]["end_time"],
                "shots": [shots[i]]
            }
    segments.append(current_segment)
    
    # Định dạng lại dữ liệu Segment theo Schema mới
    for seg in segments:
        seg["segment_caption"] = " ".join([s["caption"] for s in seg["shots"]])
        seg["object"] = []
        seg["ocr"] = []
        seg["speech"] = []
        seg["keyframe"] = []
        # Xóa mảng "shots" để JSON gọn gàng, tập trung hoàn toàn vào Segment
        seg.pop("shots", None) 
            
    return segments

# --- LUỒNG CHẠY CHÍNH ---
def process_all_videos():
    video_paths = []
    for root, dirs, files in os.walk(DRIVE_INPUT_FOLDER):
        for f in files:
            if f.endswith(('.mp4', '.avi', '.mkv')):
                video_paths.append(os.path.join(root, f))
                
    if not video_paths:
        print(f"⚠️ Không tìm thấy video nào trên Drive '{DRIVE_INPUT_FOLDER}'.")
        return

    for drive_video_path in video_paths:
        video_file = os.path.basename(drive_video_path)
        video_id = os.path.splitext(video_file)[0]
        
        folder_chua_vid = os.path.basename(os.path.dirname(drive_video_path))
        if folder_chua_vid == os.path.basename(DRIVE_INPUT_FOLDER):
            folder_chua_vid = "unknown_batch"
            
        print(f"\n[{folder_chua_vid} / {video_file}] Đang bắt đầu xử lý...")
        local_video_path = os.path.join(LOCAL_TEMP_FOLDER, video_file)
        
        print(" ⏳ Đang copy video xuống SSD để tối ưu tốc độ...")
        shutil.copy2(drive_video_path, local_video_path)
        
        kf_meta_dir = os.path.join(DRIVE_KEYFRAMES_META_FOLDER, folder_chua_vid, f"{video_id}_keyframes")
        os.makedirs(kf_meta_dir, exist_ok=True)
        
        try:
            print("  -> Cắt shot bằng TransNetV2...")
            _, single_frame_predictions, _ = transnet.predict_video(local_video_path)
            scenes = transnet.predictions_to_scenes(single_frame_predictions)
            
            print(f"  -> Trích xuất ảnh đại diện ({len(scenes)} shots) để Florence đọc...")
            keyframes, fps = extract_keyframes_from_shots(local_video_path, scenes)
            
            print("  -> Sinh caption bằng Florence-2...")
            shots_data = []
            for kf in keyframes: 
                caption = generate_caption(kf["image"])
                shots_data.append({
                    "start_time": kf["start_time"], "end_time": kf["end_time"],
                    "caption": caption
                })
                
            print("  -> Gom cụm segment (Threshold = 0.4)...")
            segments_data = group_shots_to_segments(shots_data, threshold=0.4)
            
            print("  -> 📸 Đang trích xuất Keyframes từ Segments...")
            target_frames_info = []
            
            for seg in segments_data:
                start_frame = int(seg['start_time'] * fps)
                end_frame = int(seg['end_time'] * fps)
                total_frames = end_frame - start_frame
                
                if total_frames <= 0: continue
                
                num_frames = max(3, math.ceil(total_frames / 20.0) + 1)
                step = total_frames // (num_frames - 1) if num_frames > 1 else 0
                
                for i in range(num_frames):
                    f_idx = min(start_frame + i * step, end_frame)
                    target_frames_info.append({
                        'frame_idx': f_idx,
                        'seg_id': seg['segment_id'],
                        'seg_ref': seg
                    })
                    
            target_frames_info.sort(key=lambda x: x['frame_idx'])
            
            cap = cv2.VideoCapture(local_video_path)
            current_frame = 0
            target_idx = 0
            total_targets = len(target_frames_info)
            
            while cap.isOpened() and target_idx < total_targets:
                ret, frame = cap.read()
                if not ret: break
                
                while target_idx < total_targets and current_frame == target_frames_info[target_idx]['frame_idx']:
                    info = target_frames_info[target_idx]
                    seg_id_clean = info['seg_id'].replace("_", "")
                    img_name = f"{video_id}_{seg_id_clean}_{current_frame:05d}.jpg"
                    img_path = os.path.join(kf_meta_dir, img_name)
                    
                    cv2.imwrite(img_path, frame)
                    
                    # Chỉ lưu đúng TÊN FILE vào list theo format bạn yêu cầu
                    info['seg_ref']['keyframe'].append(img_name)
                    target_idx += 1
                    
                current_frame += 1
            cap.release()
            print(f"     ✅ Đã cắt và lưu thành công {target_idx} frames hoàn hảo!")
            
            print("  -> Lưu kết quả JSON Metadata...")
            
            # --- CẬP NHẬT CẤU TRÚC JSON GỐC (ROOT) TẠI ĐÂY ---
            # Sử dụng thư mục chuẩn hóa , ở đây lưu relative path
            video_rel_path = os.path.join("videos", folder_chua_vid, video_file).replace("\\", "/")
            
            final_data = {
                "video_id": video_id,
                "type": "video",
                "video_path": video_rel_path,
                "fps": round(fps, 2),
                "segments": segments_data
            }
            
            out_json_dir = os.path.join(DRIVE_OUTPUT_FOLDER, folder_chua_vid)
            os.makedirs(out_json_dir, exist_ok=True)
            output_json_path = os.path.join(out_json_dir, f"{video_id}.json")
            
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(final_data, f, ensure_ascii=False, indent=4)
                
            print(f"🎉 Hoàn tất! File metadata đã được lưu: {output_json_path}")

        except Exception as e:
            print(f"❌ Lỗi khi xử lý {video_file}: {e}")
            
        finally:
            # 1. Dọn rác ổ cứng
            if os.path.exists(local_video_path):
                os.remove(local_video_path)
                print(" 🧹 Đã dọn dẹp file video tạm trên SSD.")
                
            # 2. Ép Python thu gom các biến/tensor rác không dùng đến trong RAM
            gc.collect() 
            
            # 3. Ép PyTorch trả lại toàn bộ VRAM trống rỗng cho Card màn hình
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            print(" ♻️ Đã reset và giải phóng hoàn toàn VRAM & RAM. Sẵn sàng cho video tiếp theo!\n" + "="*50)

if __name__ == "__main__":
    process_all_videos()