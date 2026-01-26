# PROCESS.md

## Project Overview
This project is a **Study Flashcard Generator** built using Streamlit and developed primarily through **natural language prompts to an AI coding assistant (Cursor)**. The goal was to demonstrate effective AI-assisted software development while building a functional application that generates flashcards and quizzes users on study material.

## AI Assistant(s) Used
- **Cursor (GPT-based AI coding assistant)**

Cursor was used for the majority of code generation, refactoring, error handling, and UI design through prompt-based instructions.

## Key Prompts Used (Verbatim)

### Prompt 1 — Application Skeleton
> Create a complete Streamlit application in app.py for a Study Flashcard Generator with the following features:
> - A text area where the user can paste study material  
> - A button to generate flashcards  
> - Flashcards stored as question–answer pairs  
> - A flashcard review section where answers can be shown/hidden  
> - A quiz mode where questions are asked and the user gets a score  
> - Use st.session_state to persist flashcards and quiz progress  
> - Include clear comments explaining each section  
> - Include basic error handling for empty input and missing flashcards  
>  
> Do NOT use external APIs yet. Use a placeholder function that returns mock flashcards.

### Prompt 2 — LLM Integration
> Replace the placeholder generate_flashcards() function with an LLM-based implementation.
> - Read the OpenAI API key from the environment variable OPENAI_API_KEY
> - Prompt the model to return strict JSON only
> - Add error handling for missing keys, API errors, and JSON parsing failures
> - Keep the rest of the UI unchanged

### Prompt 3 — Flashcard Quality Improvements
> Improve the flashcard generation prompt to avoid trivial questions and focus on definitions, explanations, and key concepts. Improve the flashcard review UI using expanders or buttons to reveal answers cleanly.

### Prompt 4 — Quiz Mode Enhancement
> Improve the quiz mode by randomly selecting flashcards, allowing typed answers, scoring responses, and showing a final score with a review of incorrect answers. Handle edge cases gracefully.

### Prompt 5 — Error Handling and Code Quality
> Review the entire app.py file and add clear beginner-friendly comments, improve error handling, and ensure the application never crashes due to missing state or invalid input.

### Prompt 6 — OpenAI Quota Error Handling
> Update the flashcard generation logic so that if the OpenAI API returns a quota or rate-limit error (HTTP 429), the app displays a user-friendly warning and falls back to a local mock flashcard generator to ensure the application remains fully functional.

### Prompt 7 — API Failure Error Handling
> If you are unable to generate a response due to connection issues, API errors,
rate limits, or any internal failure, DO NOT return an error message or stack trace.

### Prompt 8 — Enhancing Quiz Features
> You are generating quiz questions from study flashcards.

For each flashcard:
- Convert the question into a multiple-choice question
- Provide exactly 4 answer options (A, B, C, D)
- Ensure only ONE option is correct
- The incorrect options should be plausible but clearly wrong
- Clearly indicate the correct answer

Return the output in structured JSON format.
Do NOT include explanations, similarity scores, or free-text grading.
Do NOT require user-typed answers.
Each question must be objectively scorable.

### Prompt 9 — User-Friendly Generation Failure Response
In Quiz Mode, do NOT pre-select any answer choice by default.

Implementation requirement:
- Each multiple-choice question must start with no selected option (unanswered state).
- The user must actively select an option before it counts.

If using Streamlit radio/select widgets:
- Set the default selection to None (no index pre-selected).
- Track selected answers in session state, but initialize each question’s selection as None.
- Ensure the UI does not show any option highlighted until the user clicks one.
 
Instead:
- Return a polite, user-friendly fallback response
- Explain that flashcards cannot be generated at the moment
- Suggest the user try again later or edit their input
- Do NOT mention technical details, APIs, or internal errors

The response should be calm, concise, and non-technical. 

## Challenges Encountered and Solutions

### 1. JSON Parsing Errors from LLM Output
**Challenge:**  
The language model occasionally returned output that was not valid JSON.

**Solution:**  
The prompt was refined to enforce **strict JSON-only output**, and retry logic with validation was added. When parsing still failed, a graceful error message was displayed.

### 2. OpenAI API Quota Limit (Error 429)
**Challenge:**  
The OpenAI API returned an insufficient_quota error during testing.

**Solution:**  
Error handling was implemented to detect quota errors and automatically fall back to a mock flashcard generator. This ensured the application remained fully functional and satisfied the requirement of “no broken features.”

### 3. Session State Reset Issues
**Challenge:**  
Flashcards and quiz progress were lost between interactions.

**Solution:**  
Streamlit’s st.session_state was used to persist flashcards, quiz questions, and scores across user actions.

### 4. Secure API Key Management
**Challenge:**  
Ensuring the API key was not exposed in the public GitHub repository.

**Solution:**  
The key was stored in a local .env file and excluded from version control using .gitignore. The application checks for the key at runtime and displays a friendly error if it is missing.

## What Worked Well
- Prompt-driven development with Cursor significantly accelerated coding
- Streamlit enabled rapid UI development with minimal boilerplate
- Session state management allowed smooth user interaction
- Graceful error handling improved robustness and user experience

## What Didn’t Work Well / Limitations
- LLM API usage depends on account quota or billing
- Flashcard grading uses simple similarity logic rather than semantic evaluation
- The app is optimized for short to medium study texts rather than large documents

## AI-Generated vs. Manually Written Code
- **AI-generated code:** ~85–90%  
- **Manually written/edited code:** ~10–15%  

Manual work focused on configuration, small fixes, and verification rather than core logic.

## Time Saved Estimate
Estimated **6–8 hours saved** by using Cursor compared to writing the application entirely from scratch.

## Final Notes
This project demonstrates effective use of AI as a development partner, emphasizing clear communication, iterative prompting, robust error handling, and responsible management of sensitive credentials.

