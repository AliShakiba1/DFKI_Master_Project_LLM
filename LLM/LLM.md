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