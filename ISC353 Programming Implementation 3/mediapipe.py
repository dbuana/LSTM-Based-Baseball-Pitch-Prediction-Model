# %%
#==Importing Necessary Packages==
from pathlib import Path
import json, time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

"""
Class: ISC353 Topics in Information Science
Group Members: Mateo Henriquez (251722), Jung Hyun Park (282806), Davian Buana (271706) 

Title: LSTM-Based Prediction of Baseball Pitch Type from Pre-Pitch Motion

Hypothesis: We can predict the type of pitch a pitcher will throw based on their pre-pitch motion.

Link to one extracted pitching video: https://drive.google.com/file/d/16MW6TUNLL7Ah-v8svgnq81TNZjLeYlTm/view?usp=sharing
"""


# === Setting Paths ===
IN_DIR = Path(r"C:\Users\paul\Desktop\Topics in Information Science\pitches\2020_Pitches\modified")             # Original Video
OUT_VID_DIR  = Path(r"C:\Users\paul\Desktop\Topics in Information Science\pitches\2020_Pitches\pose")           # Pose-Estimated Video
OUT_CSV_DIR  = Path(r"C:\Users\paul\Desktop\Topics in Information Science\pitches\2020_Pitches\csv")             # CSV Data
OUT_META_DIR = Path(r"C:\Users\paul\Desktop\Topics in Information Science\pitches\2020_Pitches\meta")             # Metadata

for p in [OUT_VID_DIR, OUT_CSV_DIR, OUT_META_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Setting Processable File Extensions
EXTS = (".mp4", ".mov", ".avi", ".mkv")

# Preventing OpenCV thread Overuse
try:
    cv2.setNumThreads(0)
except Exception:
    pass


# %%


# %%


# %%


# %%
def process_one_video(
    src_path: Path,              #Path of the input video
    out_vid_dir: Path,          #Path of the Output video
    out_csv_dir: Path,          #Path of the Output CSV data
    out_meta_dir: Path,         #Path of the Output Metadata
    start_sec: float = 0.0,     
    end_sec:   float | None = None,
    model_complexity: int = 1,  # Complexity of the Mediapipe pose model, 0(Lowest)~2(Highest)
    min_detection_confidence: float = 0.5, #minimum confidence when detecting human pose initially
    min_tracking_confidence:  float = 0.5, #minimum confidence when tracing human pose
):
    t0 = time.time()
    cap = cv2.VideoCapture(str(src_path)) #open the input video file through OpenCV VideoCapture
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {src_path}") #if it hasn't opened properly, an error is going to be occured

    # reads the basic information of the input video
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0 #reads the Frame Per Second of the input video
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #reads the Width(by pixel) of the input video
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #reads the height
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None #reads the total frame of the input video
    dur_s = total / fps if (total and fps) else None #calculate them into Second

    #Set the time range of the video that the pose estimation will be conducted(in this case, this part of the code is conducting pose estimation
    #on the entire video, as cutting the video was already done by the other method
    if start_sec and start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    if end_sec is not None and dur_s is not None:
        end_ms = min(end_sec, dur_s) * 1000
    else:
        end_ms = None

    # output path, where and how the output files are going to be saved
    stem = src_path.stem
    out_vid = out_vid_dir / f"{stem}_pose.mp4"
    out_csv = out_csv_dir / f"{stem}_pose.csv"
    out_meta = out_meta_dir / f"{stem}_meta.json"

    # managing the settings for the 'Videowriter', which saves the sequence of the frames as a format of .mp4 file
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # It'll be conducted on .mp4 format, as the videos are all transformed into mp4
    writer = cv2.VideoWriter(str(out_vid), fourcc, fps, (w, h))
    if not writer.isOpened(): #check if the 'Videowriter' is opened properly
        cap.release()
        raise RuntimeError("VideoWriter open failed. Try a different folder or check codec.")

    rows = []  # CSV the list where the pose data of each frame will be stored
    frame_idx = 0 #a variable that'll be used to count the number of frame

    #Settings for the MediaPipe Pose Estimator
    with mp_pose.Pose(
        static_image_mode=False, #since it'll estimate the 'video'
        model_complexity=model_complexity, #we have already set this above
        enable_segmentation=False, #no segmentatin is needed, since the data we need is CSV file, not the video itself
        min_detection_confidence=min_detection_confidence, #already set above
        min_tracking_confidence=min_tracking_confidence,   #already set above
        smooth_landmarks=True, #I'll let mediapipe to smoothen the movement of the joints between the frame. irrevalent to the data, it's for the video
    ) as pose:

        # when the pose estimation is going on, this part of the code is written to visually show the progress of the estimation to the user
        pbar_total = (int((end_ms/1000 - start_sec) * fps) if (end_ms and fps) else total) or 0
        pbar = tqdm(total=pbar_total, desc=f"[POSE] {src_path.name}", unit="f", disable=(pbar_total==0))

        while True: #if we have manually choose where to end the pose estimation. We are doing it to the end, so this part will not work
            if end_ms is not None:
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_ms >= end_ms:
                    break

            ok, frame_bgr = cap.read() #read the next frame, if there is no next frame, that means the end of the video, therefore, ends estimation
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) #read the RGB values of the frame's GBR data
            result = pose.process(frame_rgb) #extract joints from this frame

            # 그리기
            if result.pose_landmarks: #if any joint is extracted on this frame, display them on the frame
                mp_drawing.draw_landmarks(
                    frame_bgr, #frame to display joints on
                    result.pose_landmarks, #coordinates of each data
                    mp_pose.POSE_CONNECTIONS, #data required to connect visible lines between the joints, basically about style, not data
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(), #also, about styles
                )

                # collect the coordinate data of each frame and sort them in the CSV file
                t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                for lid, lm in enumerate(result.pose_landmarks.landmark):
                    rows.append({ #sorting data on the CSV
                        "frame": frame_idx,
                        "time_ms": int(t_ms),
                        "landmark_id": lid,
                        "x": lm.x, "y": lm.y, "z": lm.z,
                        "visibility": lm.visibility
                    })

            else: # if no joint was found on this frame, NaN data will be put into the CSV cells, however, this will be later processed and 
                t_ms = cap.get(cv2.CAP_PROP_POS_MSEC) # eliminated from the dataset
                for lid in range(33):  
                    rows.append({
                    "frame": frame_idx,
                    "time_ms": int(t_ms),
                    "landmark_id": lid,
                    "x": np.nan, "y": np.nan, "z": np.nan, "visibility": np.nan
                    })

            writer.write(frame_bgr)
            frame_idx += 1
            pbar.update(1)

        pbar.close()

    cap.release()
    writer.release()

    # Save files as CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # storing metadata // metadatas of each frame are not specficially used on out report,
    meta = {
        "source": str(src_path),
        "output_video": str(out_vid),
        "output_csv": str(out_csv),
        "width": w, "height": h, "fps": fps,
        "frames_written": frame_idx,
        "duration_s_est": (frame_idx / fps) if fps else None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_complexity": model_complexity,
        "min_detection_confidence": min_detection_confidence,
        "min_tracking_confidence": min_tracking_confidence,
        "mp_pose_version": mp.__version__,
        "opencv_version": cv2.__version__,
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return out_vid, out_csv, out_meta


# %%
#Codes below, connect each frame, as well as joints and skeleton character, as a video. also,
#they're just written to visualize our data, these codse are not specifially affecting data

targets = [p for p in IN_DIR.iterdir() if p.suffix.lower() in EXTS and p.is_file()]
print(f"Found {len(targets)} video(s).")

ok, fail = 0, 0
for src in targets:
    try:
        out_vid, out_csv, out_meta = process_one_video(
            src, OUT_VID_DIR, OUT_CSV_DIR, OUT_META_DIR,
        )
        print(f"[OK]  {src.name} → {Path(out_vid).name}, {Path(out_csv).name}")
        ok += 1
    except Exception as e:
        print(f"[ERR] {src.name} : {e}")
        fail += 1

print(f"ALL DONE. ok={ok}, err={fail}")


# %%


# %%


# %%



