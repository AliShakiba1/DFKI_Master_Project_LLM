# Video Processing and Object Detection Pipeline Documentation

*Note: To see the instructions to work with this pipeline, please go to Section 6.*

## Overview
This documentation describes a Python-based video processing pipeline designed to extract synchronized frames from video recordings, run object detection using YOLO-World, and perform data normalization. The pipeline also includes an experimental module.

## Dependencies
To run this codebase, the following libraries are required:
*   `opencv-python` (`cv2`)
*   `ultralytics` (specifically for `YOLOWorld`)
*   `torch`
*   `numpy`
*   `matplotlib`

---

## Pipeline Stages

### 1. Video Configuration & Setup
The pipeline begins by configuring the input video sources with specific start and end offsets (in seconds) to trim irrelevant footage.

*   **Function:** `setup_video_capture_with_end(config)`
*   **Description:** Iterates over the video configuration dictionary, opens the video streams using OpenCV (`cv2.VideoCapture`), calculates the FPS and total frames, and determines the actual usable duration of the video after applying the start and end offsets.
*   **Outputs:** Returns an `info` dictionary containing the video capture object, FPS, start offset, and usable duration for each camera.

### 2. Video Chunking
The video is sliced into manageable, synchronized temporal blocks.

*   **Function:** `process_chunks(video_info, chunk_duration=3)`
*   **Description:** Slices the video into synchronized blocks (defaulting to 3-second chunks). For each block, it calculates the target frame index based on the camera's FPS and start offset, and extracts a single snapshot frame.
*   **Outputs:** A list of dictionaries representing each chunk. Each dictionary contains timing metadata, processed frame indices, and empty placeholders for `objects` and `action_probs`.

### 3. Object Detection (YOLO-World)
The pipeline utilizes the `YOLOWorld` model to detect specific items within the extracted frames.

*   **Setup:** Initializes the `yolov8l-worldv2.pt` model and sets a custom list of detectable classes (e.g., "measuring tape", "paper shredder", "scissors", "keyboard", "laptop").
*   **Function:** `process_objects_gopro(processed_data, video_info, model, conf_val=0.12)`
*   **Description:** Iterates through the chunked data, seeks to the pre-calculated target frame for the GoPro camera, and runs YOLO object detection with a default confidence threshold of 0.12. 
*   **Outputs:** Populates the `objects` array in the chunk dictionaries with a deduplicated list of detected class names.

### 4. Data Normalization
To ensure consistency, raw object detection labels are mapped to a standardized set of uppercase categories.

*   **Function:** `normalize_chunk_data(processed_data)`
*   **Description:** Iterates over detected objects and uses keyword matching to group variations into standard categories. For example, "metal scissors" and "shears" map to `SCISSORS`; "screen" maps to `MONITOR`.
*   **Outputs:** Updates the chunk data with normalized object lists.

### 5. Data Persistence
The pipeline includes a utility for saving and loading the processed data state.

*   **Function:** `save_load(path, save=False, config=None)`
*   **Description:** Saves the current state of the pipeline (the list of chunk dictionaries) to a JSON file, or loads it from disk if `save=False`.

---

### 6. Pipeline Instructions


Here is the hardware configuration formatted cleanly in Markdown, including the note that this is the system you are running your model on:

---

### Server Hardware Configuration

Below is the system hardware environment on which the model is running:

> **Note:** I also add `%%time` before each run to estimate the execution time.

| Component | Specifications |
| --- | --- |
| **CPU** | Intel® Xeon® Gold 6354 CPU @ 3.00GHz |
| **CPU Cores** | 72 |
| **RAM** | 1.0 TiB |
| **GPU** | NVIDIA A2 (15,356 MiB) |


> **Recommendation:** It is highly recommended to keep the maximum length of each video for each task **under 10 minutes**. The optimal duration is **5–7 minutes**. Longer videos require a much larger `custom_classes` list, which increases detection noise and reduces the overall quality of the YOLO-World results.

#### Step 1 — Configure the video

Before running the pipeline, define the video path and remove the unimportant parts of the recording by setting the start and end offsets (in **seconds**).

```python
video_paths = {
    "gopro": {
        "path": "../../gopro/g_joanna.MP4",
        "start_offset": 30,
        "end_offset": 5
    }
}
```

- `path`: Location of the input video.
- `start_offset`: Seconds to skip from the beginning.
- `end_offset`: Seconds to ignore from the end.

After configuring the video, **run the notebook code blocks one by one** until the **Install YOLOWorld** section is completed.

#### Step 2 — Create `custom_classes`

For every new work/task, create a dedicated `custom_classes` list.

Since this pipeline does **not** use deep learning for activity recognition, it only detects and tracks the objects visible in each frame. Therefore, include **multiple variations and synonyms** of every object that may appear in the scene.

For example:

```python
custom_classes = [
    "scissors",
    "metal scissors",
    "shears",
    "measuring tape",
    "ruler",
    "keyboard",
    "laptop",
    "monitor"
]
```

A richer class list improves coverage, but adding too many unrelated classes can introduce noise. Keep the list focused on the current task.

#### Step 3 — Test the detection

After defining `custom_classes`, run the testing code blocks on **a few short sections of the video**.

Verify that:

- All important objects are detected.
- Incorrect detections are minimal.
- The selected classes match the objects in your workspace.

If necessary, modify `custom_classes` and test again before processing the full video.

#### Step 4 — Process and save

Once the detection quality is satisfactory, continue running the remaining code blocks **one by one** until reaching the `save_load` section.

Use `save_load` to save the processed chunk data as a JSON file for later use.

```python
save_load("output_chunks.json", save=True, config=processed_data)
```
