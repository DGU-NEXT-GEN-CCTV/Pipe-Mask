import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.optimize import linear_sum_assignment
from ultralytics.data.annotator import auto_annotate
from utils.logger import Logger

logger = Logger()

def parse_args():
    logger.log("[bold] ‣ Initializing... [/bold]")
    parser = ArgumentParser()
    parser.add_argument('--input-dir', type=str, default="data/input")
    parser.add_argument('--output-dir', type=str, default="data/output")
    parser.add_argument('--working-dir', type=str, default="data/working")
    parser.add_argument('--det-model', type=str, default="model/yolo11x.pt")
    parser.add_argument('--sam-model', type=str, default="model/sam2.1_b.pt")
    args = parser.parse_args()
    
    logger.console_args(vars(args))
    
    return args

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
    
    logger.log(f"[bold] ‣ Loading frames and segmentations from {frame_dir} and {seg_dir}... [/bold]")
    
    frame_list = sorted(os.listdir(frame_dir))
    seg_list = sorted(os.listdir(seg_dir))

    frame_seg_list = []

    for f, s in zip(frame_list, seg_list):
        filename = os.path.splitext(f)[0]
        frame = Image.open(os.path.join(frame_dir, f))
        seg = parse_seg(os.path.join(seg_dir, s), frame.size)
        frame_seg_list.append((filename, frame, seg))
        
    logger.log(f"[bold green] ∙ Loaded {len(frame_seg_list)} frames and segmentations [/bold green]")
        
    return frame_seg_list


def extract_frames(video_path: str, working_dir: str):
    logger.log(f"[bold] ‣ Extracting frames from {video_path}... [/bold]")
    
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
    
    logger.log(f"[bold green] ∙ Extracted {frame_count} frames [/bold green]")
    
    return video_name, frame_list, cur_working_dir, cur_frame_dir, cur_masked_frame_dir, cur_seg_dir


def masking(frame_seg_list: list, cur_masked_frame_dir: str):
    logger.log(f"[bold] ‣ Masking frames... [/bold]")
    
    if not os.path.exists(cur_masked_frame_dir):
        os.makedirs(cur_masked_frame_dir, exist_ok=True)
        
    pre_seg = None
    
    masked_dir = []

    def get_centroid(contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return tuple(contour.mean(axis=0)[0].astype(int))
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    for filename, frame, seg in tqdm(frame_seg_list, desc="Masking frames"):
        if pre_seg is not None and len(seg) > 0 and len(pre_seg) > 0:
            shape_weight = 0.7
            distance_weight = 0.3
            max_dist = np.sqrt(frame.size[0]**2 + frame.size[1]**2)

            centroids_seg = [get_centroid(s) for s in seg]
            centroids_pre_seg = [get_centroid(s) for s in pre_seg]

            cost_matrix = np.full((len(seg), len(pre_seg)), np.inf)
            
            for i, s1 in enumerate(seg):
                for j, s2 in enumerate(pre_seg):
                    shape_cost = cv2.matchShapes(s1, s2, cv2.CONTOURS_MATCH_I1, 0.0)
                    
                    dist = np.linalg.norm(np.array(centroids_seg[i]) - np.array(centroids_pre_seg[j]))
                    dist_cost = dist / max_dist
                    
                    cost_matrix[i, j] = (shape_weight * shape_cost) + (distance_weight * dist_cost)
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            new_seg = [None] * len(pre_seg)
            matched_new_indices = set()

            for r, c in zip(row_ind, col_ind):
                # A threshold to prevent matching dissimilar objects
                if cost_matrix[r, c] < 0.5:
                    new_seg[c] = seg[r]
                    matched_new_indices.add(r)
            
            unmatched_new_segs = [s for i, s in enumerate(seg) if i not in matched_new_indices]
            
            # Fill empty slots with unmatched new segments or append them
            final_seg = []
            unmatched_idx = 0
            for s in new_seg:
                if s is not None:
                    final_seg.append(s)
                elif unmatched_idx < len(unmatched_new_segs):
                    final_seg.append(unmatched_new_segs[unmatched_idx])
                    unmatched_idx += 1
            
            # Add any remaining unmatched segments
            if unmatched_idx < len(unmatched_new_segs):
                final_seg.extend(unmatched_new_segs[unmatched_idx:])

            seg = final_seg
                
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
        
    logger.log(f"[bold green] ∙ Processed {filename} with {len(seg)} segments [/bold green]")

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
    logger.console_banner()
    args = parse_args()
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.working_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    video_list = load_video_list(args.input_dir)
    
    for video_path in video_list:
        logger.console_banner()
        logger.log(f"Processing video: {video_path}")
        video_name, frame_list, cur_working_dir, cur_frame_dir, cur_masked_frame_dir, cur_seg_dir = extract_frames(video_path, args.working_dir)

        logger.log(f"[bold] ‣ Segmenting frames... [/bold]")
        for frame_path in tqdm(frame_list, desc=f"Segmenting {video_name}"):
            
            auto_annotate(
                data=frame_path, # 프레임 이미지 경로
                det_model=args.det_model, # 객체 검출 모델
                sam_model=args.sam_model, # 세그멘테이션 모델
                classes=[0], # 사람만 세그멘테이션
                output_dir=cur_seg_dir # 라벨링 결과 저장 디렉토리
            )
        logger.log(f"[bold green] ∙ Segmentation completed for {video_name} [/bold green]")
            
        frame_seg_list = load_frame_seg(cur_frame_dir, cur_seg_dir)
        masked_dir_list = masking(frame_seg_list, cur_masked_frame_dir)
        compress_video(video_name, masked_dir_list, args.output_dir)

if __name__ == "__main__":
    main()