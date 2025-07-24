Of course. I will improve the Evaluation Plan section for you by completing the unfinished parts and structuring it with clear hierarchical notation. I will use the detailed error analysis plan from the ticket in the Canvas to flesh out the "Error Analysis" section.

Here is the raw markdown for the complete and updated evaluation plan. You can copy and paste it directly.

```markdown
# Survey Assist SIC LLM Evaluation plan

## Overview

Survey Assist (SA) is a developing prototype that aims to improve the codability and classification accuracy of Transformed Labour Force Survey (T-LFS) responses. Survey Assist can automatically generate the most relevant follow up survey questions in a live session with a respondent by making use of a Large Language Model (LLM).

The purpose of this document is to provide a framework for evaluating the Survey Assist System. The LLM is prompted to assign a SIC code and follow-up question when given the survey respondents’ answers to questions about employment.

The evaluation plan covers several aspects including:

* **Alignment with clerical coders (CC):** The extent to which outputs from Survey Assist agree with those provided by the clerical coders.
* **Error analysis:** An assessment to understand the root cause of the main errors in SA responses.
* **Model output stability:** Measuring the extent to which Survey Assist results change when model parameters change.
* **Quality of the follow-up:** An assessment of reading score, suitability and ability to disambiguate the shortlist of possibilities.
* **Bias:** A simulation of a survey using another LLM to play the part of the respondent.

---

## Metrics of Alignment

### Human coder alignment

The dataset of 2000 broadly representative selection from across all sections. Provided by Clare Cheeseman and Hannah Tomlinson, these data are a carefully selected representative sample with expert SIC allocations that will be used to determine the benchmark of performance.

#### Unambiguous responses

When a coder gives only a single assessment to five digits, he has identified unambiguously the code from the given information without need of a follow-up question. These will be used as a subset for measures.

### Scenario 1 – SA performance on unambiguously codable

**Purpose:** To give an initial score of a direct match between the LLM’s first choice and the coders’ unambiguous choice. Of the initial 2079 dataset, 1291 are considered unambiguous.

**Business need/motivation:** This gives the team a baseline of SA’s ability to match the coders’ assessment, when the coder is sure of the answer.

**Approach:** A measure of the number of responses where the first choices are the same, given as a percentage. This is done at both all five digits and again for only the first two digits (SIC Division).

**How the metric will be used:** This will be used to give an initial measure of Survey Assist’s direct accuracy. It will represent a minimum performance level against which proposed changes must improve upon before acceptance.

### Scenario 2 – Any shortlist match rate to unambiguously codable

**Purpose:** Survey Assist provides multiple answers as a shortlist. This metric checks if any of the SA answers match the coders’ selection for the unambiguous data.

**Business need/motivation:** This gives the team an insight as to whether the LLM is at least in the appropriate area of judgement with the human coders.

**Approach:** A count of the number of responses where one or more SA responses matched the coders’ answer. This is done at both five digits and again for only the first two digits.

**How the metric will be used:** This will be used to check for minimum LLM adherence to the prompt and ability to make use of the RAG provided dataset. This is a ‘canary’ test and should always give a high result.

### Scenario 3 – Top Clerical Coder response featuring in SA shortlist rate

**Purpose:** This metric makes use of the coders’ tendency to put the preferred option into the first choice. The rate at which this code features in the SA shortlist is a valuable metric.

**Business need/motivation:** This provides a more nuanced measure of SA’s ability to consider similar options to the coder. It will provide a basis for the follow-up question.

**Approach:** A count of the number of responses where one of the SA responses matches the coders’ first choice. This is done at both five digits and two digits for both the unambiguous subset and the full dataset.

**How the metric will be used:** This will be used to provide a basis for prompt improvement and follow-up question relevance.

### Scenario 4 – Clerical Coder Shortlist and SA shortlist containing at least one common element

**Purpose:** This metric measures the overlap between the full set of human codes and the full set of model suggestions.

**Business need/motivation:** This provides a broad measure of alignment. A high score indicates that the model and human are thinking along similar lines, even if the top choices don't match. A follow-up question will be required, so absolute accuracy isn’t as important at this stage.

**Approach:** A count of the number of responses where at least one of the SA responses is found in the CC’s shortlist. This is done at both five digits and two digits.

**How the metric will be used:** This will be used to provide a basis for prompt improvement and follow-up question relevance.

### Scenario 5 – Jaccard measure

**Purpose:** To measure the quality and relevance of the entire suggestion list, not just the presence of a single correct answer.

**Business need/motivation:** A high accuracy score could still come from a noisy list (e.g., one correct code among four bad ones). The Jaccard score tells us how "clean" the suggestion list is. A high score means the model's suggestions closely mirror the human coder's choices.

**Approach:** The average Jaccard Index (Intersection over Union) is calculated between the set of model codes and the set of clerical codes for each row. This is done at both five digits and two digits.

**How the metric will be used:** This metric provides a more nuanced view of performance. It will be used to evaluate if changes to prompts are making the suggestion lists more relevant and less noisy.

### Scenario 6 – Error analysis

**Purpose:** To systematically analyse classification errors to generate data-driven hypotheses for improving system performance.

**Business need/motivation:** A single accuracy number doesn't tell us *what kind* of mistakes the model is making. This analysis provides the qualitative insights needed to guide prompt engineering and identify potential weaknesses in the RAG system.

**Approach:** The analysis is broken down into several tasks:

* **1. Identify Systematic Classification Errors:**
    * Filter the results to create a subset of incorrect classifications.
    * Manually review a sample of these errors to categorise recurring themes (e.g., Job Role vs. Industry Confusion, Wholesale vs. Retail Confusion, Specificity Errors).
    * Verify the `initial_llm_payload` for critical errors to rule out data delivery issues.

* **2. Investigate Potential for RAG/Vector Store Improvement:**
    * Analyse "near miss" errors where the correct code was in the suggestion list but not ranked first. A high number of these suggests the RAG is working but the prompt needs refinement.
    * Review the AI's "reasoning" text for phrases indicating uncertainty, which may point to generic context from the RAG.

* **3. Compare Model Performance (e.g., Gemini 1.5 vs. 2.0 Flash):**
    * Segment the error data by model or prompt.
    * Compare the distribution of error categories between models to understand if issues are prompt-specific or model-specific.

* **4. Develop an Automated Metric for Error Severity:**
    * Implement a "Hierarchy Mismatch Score" (0-5) that quantifies how "wrong" a prediction is by measuring its distance from the true code within the SIC hierarchy (e.g., a score of 1 means a 4-digit match, a score of 5 means no match at any level).
    * Apply this metric to the error dataset to distinguish between fine-grained errors and fundamental classification mistakes.

**How the metric will be used:** The findings will be used to produce a summary report detailing the top error patterns and to generate specific, actionable hypotheses for improving the prompt templates and RAG system.

### Scenario 7 – Model Stability

**Purpose:** To check the consistency of the system’s performance as prompts and models change during development.

**Business need/motivation:** To provide a quick, reliable way to ensure that incremental changes are genuinely improving the system and not causing regressions in performance.

**Approach:** A representative subset of the data will be designated as a "benchmark set." After any significant change to a prompt or model, this benchmark set will be re-run through the system. Key metrics (e.g., 5-digit accuracy, Jaccard score, Hierarchy Mismatch Score) will be compared against the previous results.

**How the metric will be used:** This will provide a Go/No-Go signal for accepting changes. It ensures a continuous improvement cycle and prevents accidental degradation of the system's performance.
```
