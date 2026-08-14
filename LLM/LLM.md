# LLM Activity Classification Pipeline Documentation

## Overview

This documentation details a Python-based pipeline designed to classify human activities from sequential video chunks using a Large Language Model (LLM). The pipeline loads a lightweight instruction-tuned model, formats synchronized object and audio data, and prompts the model to output strict, structured JSON containing low-level actions and high-level tasks.

## Dependencies

The following libraries are required to run this pipeline:

* `transformers`

* `accelerate`

* `torch`

* Standard Python libraries: `json`, `gc`, `os`, `re`


---

## Pipeline Stages

### 1. Model Initialization

The system is configured to use a specific, lightweight LLM for inference.

* **Model Selection:** While other models (like DeepSeek-R1-Distill-Qwen-1.5B or Qwen2.5 variants) are listed as options, the pipeline defaults to `microsoft/Phi-3.5-mini-instruct`, which is noted as being better suited for this specific problem.


* **Configuration:** The model is loaded in `bfloat16` mode and utilizes `"sdpa"` (Scaled Dot-Product Attention) for optimized performance.


* **Memory Management:** The script utilizes `gc.collect()` and `torch.cuda.empty_cache()` to clear memory before loading the model onto a CUDA-enabled device.



### 2. Model Testing and Reasoning Extraction

Before processing the actual dataset, the pipeline includes a testing phase to verify the model's functionality and parsing logic.

* **Test Prompt:** The model is tested with a simple prompt: "In one sentence, what is a keyboard used for?".


* **Output Parsing:** The code includes logic to separate the model's internal reasoning (text generated inside `<think>` tags) from its final output response.



### 3. Prompt Engineering

A detailed system prompt (`SYSTEM_PROMPT_g`) is defined to instruct the LLM on its role as an "expert activity classifier". The prompt enforces strict constraints:

* **Allowed Verbs:** Limits low-level actions to specific categories, such as Reach (e.g., `REACH_OBJECT`), Grasp (e.g., `GRASP_OBJECT`), Pick/Place, Open/Close, Control, Tools (e.g., `CUT`, `WRITE`), Body, and Inspect. The model is strictly instructed never to invent new verbs.


* **Allowed High-Level Tasks:** Requires the model to choose exactly one high-level task per chunk from a predefined list (e.g., "typing in keyboard", "measuring the screen", "shred paper with shredder").


* **Advanced Reasoning Rules:** Instructs the model on how to handle continuous actions (indicated by the `[Propagated]` tag), infer actions from empty text using visible objects, and handle multiple actions within a single chunk.



### 4. Data Formatting

The input data must be cleaned and restructured before being fed into the LLM.

* **Data Source:** The pipeline loads the previously processed dataset from `./fd_silent_fix.json`.


* **Function (`format_chunks_optimized`):** This function prepares the data by splitting the original string-based `time` range (e.g., "0-3") into distinct `time_start` and `time_end` fields. It returns a formatted list of dictionaries containing the chunk ID, start and end times, participant text, and visible objects.



### 5. Batch Processing and Persistence

The core of the pipeline processes the formatted dataset in batches and saves the parsed JSON output.

* **Batching:** The data is processed in batches of 10 chunks at a time to manage memory and context limits.


* **Generation Parameters:** The model generates responses using greedy decoding (`do_sample=False`) to eliminate syntax errors and a `repetition_penalty` of 1.02 to prevent the model from looping on repetitive JSON keys.


* **JSON Extraction:** The script extracts only the valid JSON array from the model's text output, stripping away any reasoning text or markdown formatting (like ````json`).


* **Iterative Saving:** Extracted JSON arrays are safely loaded and appended to a persistent file named `all_processed_chunks_with_reasoning.json` after every successful batch.


* **Error Handling & Cleanup:** If a batch fails to parse as valid JSON, the error is caught, the batch index is recorded in `failed_batches`, and the script continues. After each batch, variables are deleted, and CUDA memory is emptied to prevent memory overflow.


Here is the formatted section ready to be added to your documentation. I have rewritten your notes to match the professional, clear, and structured style of the existing pipeline document.

---

### 6. Pipeline Instructions

#### Step 1 — Model Selection & Initialization

In the first code block, you have the option to select from various models. It is highly recommended to use the default `"microsoft/Phi-3.5-mini-instruct"`, as it is the most optimized for this problem. Please note that this model requires approximately **8 GB of VRAM** once loaded into memory.

#### Step 2 — Model Verification

Before running your dataset, execute the testing block. This ensures that the model has loaded correctly, the tokenizer is working, and the reasoning extraction logic is functioning properly.

#### Step 3 — System Prompt Configuration (Critical)

The `SYSTEM_PROMPT_g` definition is the most important part of the pipeline. As noted in the Narration Documentation, **you must finalize your high-level and low-level task lists here before you record your audio narration**. Make sure your prompt is completely ready to ensure the vocabulary perfectly matches your recorded text.

#### Step 4 — Batch Size and Token Limits

The `batch_size` variable controls how many chunks are sent to the model simultaneously.

* A default `batch_size` of **10** is optimal for this specific model and task complexity.
* If you decide to increase the batch size to process more chunks at once, you **must** also increase the `max_new_tokens` parameter in the generation configuration. Otherwise, the model will run out of space and the JSON output will be cut off.

#### Step 5 — Execution and JSON Extraction

Run the final processing loop to iterate through all the chunks. Because the model generates unnecessary extraneous text (such as internal `<think>` tags and markdown formatting), this loop utilizes robust parsing logic to strip away the noise and strictly isolate, validate, and save only the final JSON array.

#### Step 6 — Time Formatting Utility

Execute the final utility code block (following the main processing loop). This function converts the raw seconds output in your JSON file into a more readable `min:sec` format for easier human review.