# Audio Transcription and Text-Merging Pipeline Documentation

## Overview

This documentation describes a pipeline designed to extract audio from video files, transcribe the speech using OpenAI's Whisper model, and seamlessly merge the resulting text with existing object detection data. A key feature of this pipeline is its ability to handle periods of silence during continuous actions by propagating narration text to adjacent synchronized chunks.

## Dependencies

To execute this codebase, the following libraries and system tools are required:

* `openai-whisper`

* `pydub`

* `ffmpeg` (command-line tool used via `subprocess`)


* Standard Python libraries: `os`, `subprocess`, `wave`, `json`


---

## Pipeline Stages

### 1. Audio Extraction and Transcription

The pipeline extracts audio from a provided video and transcribes it into text, segmenting the words into distinct time windows.

* **Setup:** Initializes the Whisper model using the "turbo" (`large-v3-turbo`) version loaded onto a CUDA device for high accuracy.


* **Function:** `transcribe_windows(video_path, model_whisper, chunk_length_sec=3.0, start_offset=0.0, end_offset=0.0)`

* **Description:**
* Extracts audio from the video into a temporary `.wav` file using an `ffmpeg` command.


* Calculates the usable audio duration by applying start and end offsets.


* Transcribes the audio using strict anti-hallucination and anti-repetition settings (`condition_on_previous_text=False`, `no_speech_threshold=0.6`, `logprob_threshold=-1.0`, `compression_ratio_threshold=2.4`).


* Filters out background silence and maps the detected words into non-overlapping 3-second chunks based on exact timestamps.




* **Outputs:** Returns a list of dictionaries containing the `start` time, `end` time, and the mapped `text` for each chunk.



### 2. Data Merging

The transcribed narration text is combined with an existing dataset containing object detection information.

* **Function:** `merge_and_clean_video_text(video_list, text_list)`

* **Description:** Iterates through the video object detection chunks and removes unnecessary metadata, specifically `cameras_data` and `action_probs`. It then safely matches the corresponding transcription text from the Whisper output by index, labeling it under the key `participant_text`.


* **Outputs:** Returns a combined list of dictionaries containing synchronized object detection labels and participant narration.



### 3. Narration Propagation

To account for continuous actions where a participant may only speak at the beginning or end of the task, the pipeline duplicates relevant text into adjacent empty chunks.

* **Function:** `propagate_narrations(processed_data, max_window=2)`

* **Description:**
* Executes a forward pass to propagate speech into following silent chunks (up to a default maximum window of 2, or roughly 6 seconds) if there is an overlap in detected objects.


* Executes a backward pass to propagate speech into preceding silent chunks using the same object-continuity logic.


* Appends the tag `[Propagated]` to any text that was copied over to distinguish it from originally spoken words.




* **Outputs:** Updates the chunk data with the newly propagated text to ensure continuous context.



### 4. Data Persistence

The system utilizes a utility function to save and load the merged, propagated JSON data.

* **Function:** `save_load(path, config=None, save=False)`

* **Description:** Loads the initial normalized object detection data and, once processing is complete, saves the final compiled dataset (e.g., `fd_silent_fix.json`) securely to the disk.



Here is the added Section 5 formatted to match the style of the pipeline documentation.

---

### 5. Pipeline Instructions

#### Step 1 — Configure the Transcription

To begin the transcription process, you must configure the pipeline parameters to exactly match the ones used in the Video Object Detection pipeline.

```python
# Call the function directly with your video parameters
chunks_transcript = transcribe_windows(
    video_path="./j_t_by_ali.m4a",
    model_whisper=model_whisper,
    chunk_length_sec=3.0,
    start_offset=17.0,
    end_offset=1.0
)

```

* **`video_path`**: Can be a video or audio file.
* **`start_offset` & `end_offset**`: Must perfectly align with the video setup to ensure temporal synchronization.
* **`chunk_length_sec`**: Must remain at `3.0` to match the visual frames.

Once configured, run the code blocks one by one until the end of the file.

#### Step 2 — Narration Recording Guidelines (Critical)

Because this pipeline ultimately feeds into a smaller language model (4B parameters) to optimize for speed and hardware constraints, the resulting output is highly sensitive to noise. To guarantee high-quality results, **you must adhere strictly to the following narration rules:**

* **Preparation & Environment:**
* Prepare your high-level and low-level task lists (which will be used for the LLM prompts) and keep them in front of you while recording.
* Record in an isolated, quiet location with no background noise.
* *Tip:* If recording on an iPhone, manually turn on the "Studio Voice" (Voice Isolation) feature, as it is turned off by default, to ensure clean audio.


* **Complete & Consistent Phrasing:**
* Speak every high-level activity **completely and accurately** (e.g., always say *"writing on paper"*). Do not use shortened phrases or combine two distinct tasks.
* Stick to **one specific phrase** per task type. For example, if the participant is typing, only say *"typing on keyboard"*. Do not improvise with phrases like *"writing on keyboard"*, as overlapping vocabulary confuses the small model.


* **Repetition for Continuous Actions:**
* If a participant performs the same action for a long period (e.g., writing continuously for 15 seconds), **repeat the high-level task phrase every 4 to 5 seconds**. Because the data is fed to the model in batches of 7-10 chunks, failure to repeat the action will leave chunks empty and introduce noise.


* **Include Low-Level Actions:**
* In between the main high-level actions, narrate the transitional low-level actions (e.g., *"open the box"*, *"close the box"*, *"pick up the pen"*). Ensure these perfectly match your pre-planned lists.


* **Vocal Delivery:**
* Speak very fluently, clearly, and at a steady pace to ensure the Whisper model correctly stamps the word timestamps.