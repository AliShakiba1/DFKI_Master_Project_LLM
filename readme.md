# LLM Video Understanding & Activity Prediction Pipeline

This folder contains the core machine learning and natural language processing pipeline designed to process video recordings, perform zero-shot object detection, integrate verbal participant narrations, and leverage Large Language Models (LLMs) to predict actions and analyze behavior.

> 📁 **Future Multimodal Expansion:** This repository is structured to prioritize the `LLM` video pipeline. A separate `apple_watch` folder will be added to this project later to handle wearable sensor data integration.

---

## 📁 Repository Structure & Workflow

```
LLM/
├── 1. video_oo_chuck_0.ipynb     # Video processing, object detection & hand tracking
├── 2. narration.ipynb            # Narration alignment & text context propagation
├── 3. llm.ipynb                  # LLM reasoning engine & prediction compiler
├── 4. fd_silent_fix.json         # Intermediate dataset (aligned objects + narrations)
└── 5. output_predictions.json    # Final LLM output predictions mapped by chunk ID

```

---
Here is a highly condensed, ultra-scannable version of your step-by-step file documentation:

---

## 🛠️ Step-by-Step File Documentation

### **1. Video to Object Detection (`video_oo_chuck_0.ipynb`)**

Processes raw video feeds and extracts core computer vision features.

* **Temporal Chunking:** Splits footage into uniform **3-second chunks**.


* **Efficient Seeking:** Reads only **1 frame per chunk** via fast-seeking, slashing I/O workload by **98.6%**.


* **YOLOv8-World:** Detects workspace objects using a custom, targeted vocabulary.


* **Taxonomy Normalization:** Standardizes varied text labels into clean uppercase categories (e.g., `MEASURING_TAPE`).


* **Hand Tracking:** Classifies gestures (`REACH_OBJECT`, `GRASP_OBJECT`, `ACTIVE`, `IDLE`) using **Google MediaPipe** and majority voting.



### **2. Narration Merging & Propagation (`narration.ipynb`)**

Aligns verbal participant transcripts with the visual timeline.

* **Metadata Fusion:** Syncs text descriptions directly to corresponding 3-second chunks.
* **Smart Propagation:** Automatically carries previous narratives forward into empty chunks if object or hand activity remains continuous.

### **3. LLM Reasoning (`llm.ipynb`)**

The reasoning engine that uses natural language prompts to classify human behavior.

* **Prototyping:** Runs quick inference tests on **two random chunks** to evaluate prompts.
* **Inference Pipeline:** Runs a batch loop to generate activity classifications for the entire timeline.

### **4. Integrated Dataset (`fd_silent_fix.json`)**

The intermediate feature dataset containing aligned timestamps, active camera frames, detected objects, hand motion, and text narrations.

### **5. Final Predicted Activities (`output_predictions.json`)**

The final compiled output mapping each `chunk_id` to a categorical activity prediction, confidence level, and textual reasoning.


