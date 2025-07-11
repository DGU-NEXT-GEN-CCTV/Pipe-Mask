import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from ultralytics.data.annotator import auto_annotate

def load_video_list(input_dir: str) -> list:
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    video_files = sorted(video_files)
    if not video_files:
        raise ValueError(f"No video files found in the input directory '{input_dir}'.")
    
    return [os.path.join(input_dir, f) for f in video_files]


def load_frame_seg(frame_dir: str, seg_dir: str):
    def parse_seg(seg_file: str, image_size: tuple):
        w, h = image_size
        seg = []
        with open(seg_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            pos_list = line.split(' ')
            pos_list = [float(p) for p in pos_list if p.strip()]
            contour = []
            for i, p in enumerate(pos_list):
                if i != 0 and i % 2 == 0:
                    contour.append((int(pos_list[i-1] * w), int(p * h)))
            contour = np.array(contour, dtype=np.int32)
            seg.append(contour)
        return seg
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"Frame directory '{frame_dir}' does not exist.")
    if not os.path.exists(seg_dir):
        raise FileNotFoundError(f"seg directory '{seg_dir}' does not exist.")
    
    frame_list = sorted(os.listdir(frame_dir))
    seg_list = sorted(os.listdir(seg_dir))

    frame_seg_list = []

    for f, s in zip(frame_list, seg_list):
        filename = os.path.splitext(f)[0]
        frame = Image.open(os.path.join(frame_dir, f))
        seg = parse_seg(os.path.join(seg_dir, s), frame.size)
        frame_seg_list.append((filename, frame, seg))
        
    return frame_seg_list


def extract_frames(video_path: str, working_dir: str):
    video_name = os.path.basename(video_path).split('.')[0]
    frame_list = []
    
    cur_working_dir = os.path.join(working_dir, video_name)
    cur_frame_dir = os.path.join(cur_working_dir, "frames")
    cur_masked_frame_dir = os.path.join(cur_working_dir, "masked_frames")
    cur_seg_dir = os.path.join(cur_working_dir, "segs")
    
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(cur_working_dir, exist_ok=True)
    os.makedirs(cur_frame_dir, exist_ok=True)
    os.makedirs(cur_masked_frame_dir, exist_ok=True)
    os.makedirs(cur_seg_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(cur_frame_dir, f"{frame_count:08d}.jpg")
        frame_list.append(frame_path)
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    return video_name, frame_list, cur_working_dir, cur_frame_dir, cur_masked_frame_dir, cur_seg_dir


def masking(frame_seg_list: list, cur_masked_frame_dir: str):
    if not os.path.exists(cur_masked_frame_dir):
        os.makedirs(cur_masked_frame_dir, exist_ok=True)
        
    pre_seg = None
    
    masked_dir = []

    for filename, frame, seg in tqdm(frame_seg_list, desc="Masking frames"):
        if pre_seg is not None:
            if len(seg) > 0 and len(pre_seg) > 0:
                cost_matrix = np.zeros((len(seg), len(pre_seg)))
                for i, s1 in enumerate(seg):
                    for j, s2 in enumerate(pre_seg):
                        cost_matrix[i, j] = cv2.matchShapes(s1, s2, cv2.CONTOURS_MATCH_I1, 0.0)
                
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                new_seg = [None] * len(seg)
                matched_rows = set()
                for r, c in zip(row_ind, col_ind):
                    if c < len(new_seg):
                        new_seg[c] = seg[r]
                        matched_rows.add(r)

                unmatched_seg = [s for i, s in enumerate(seg) if i not in matched_rows]
                
                for i in range(len(new_seg)):
                    if new_seg[i] is None and unmatched_seg:
                        new_seg[i] = unmatched_seg.pop(0)
                
                seg = [s for s in new_seg if s is not None]
                
        for s_idx, s in enumerate(seg):
            cur_masked_image_dir = os.path.join(cur_masked_frame_dir, str(s_idx))
            if cur_masked_image_dir not in masked_dir:
                masked_dir.append(cur_masked_image_dir)
            os.makedirs(cur_masked_image_dir, exist_ok=True)
            
            mask = np.zeros((frame.size[1], frame.size[0]), dtype=np.uint8)
            cv2.fillPoly(mask, [s], 255)
            masked_image = cv2.bitwise_and(np.array(frame), np.array(frame), mask=mask)
            masked_image = Image.fromarray(masked_image)
            masked_image.save(os.path.join(cur_masked_image_dir, f"{filename}.png"))

        pre_seg = seg

    return masked_dir


def compress_video(video_name: str, masked_dir_list: list, output_dir: str):
    if not masked_dir_list:
        raise ValueError("No masked directories provided for video compression.")
    
    for masked_dir in masked_dir_list:
        if not os.path.exists(masked_dir):
            raise FileNotFoundError(f"Masked directory '{masked_dir}' does not exist.")
        
        images = sorted(os.listdir(masked_dir))
        if not images:
            raise ValueError(f"No images found in the masked directory '{masked_dir}'.")
        
        first_image_path = os.path.join(masked_dir, images[0])
        first_image = Image.open(first_image_path)
        width, height = first_image.size

        output_path = os.path.join(output_dir, f"{video_name}_{os.path.basename(masked_dir)}.mp4")
        print(f"Creating video: {output_path}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        for image_name in tqdm(images, desc=f"Compressing {video_name}_{os.path.basename(masked_dir)}"):
            image_path = os.path.join(masked_dir, image_name)
            frame = cv2.imread(image_path)
            out.write(frame)
        
        out.release()
    print(f"Video saved to {output_path}")

def main():
    input_dir = "data/input"
    working_dir = "data/working"
    output_dir = "data/output"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    video_list = load_video_list(input_dir)
    for video_path in video_list:
        print(f"Processing video: {video_path}")
        video_name, frame_list, cur_working_dir, cur_frame_dir, cur_masked_frame_dir, cur_seg_dir = extract_frames(video_path, working_dir)
        
        for frame_path in frame_list:
            auto_annotate(
                data=frame_path, # 프레임 이미지 경로
                det_model="model/yolo11x.pt", # 객체 검출 모델
                sam_model="model/sam2.1_b.pt", # 세그멘테이션 모델
                classes=[0], # 사람만 세그멘테이션
                output_dir=cur_seg_dir # 라벨링 결과 저장 디렉토리
            )
        
        frame_seg_list = load_frame_seg(cur_frame_dir, cur_seg_dir)
        masked_dir_list = masking(frame_seg_list, cur_masked_frame_dir)
        compress_video(video_name, masked_dir_list, output_dir)
        

if __name__ == "__main__":
    main()