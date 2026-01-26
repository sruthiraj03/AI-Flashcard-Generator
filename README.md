# AI Study Flashcard Generator

A **Streamlit-based Study Flashcard Generator** built using **AI-assisted development with Cursor**.  
The application allows users to input study material, automatically generate flashcards using an LLM, and test their knowledge through an interactive multiple-choice quiz with scoring.

This project demonstrates effective use of **prompt-driven AI development**, robust **state management**, and **graceful error handling** in a production-style AI application.

## 🚀 Features

### 📄 Input Study Material
- Paste any study text (notes, definitions, concepts).
- Select the number of flashcards to generate (maximum of 20).

### 🧠 AI-Generated Flashcards
- Flashcards are generated automatically using an LLM.
- If the AI is unavailable or an issue occurs, the application displays a user-friendly fallback message instead of a technical error.

### 👀 Flashcard Review Mode
- Review all generated flashcards.
- Toggle between questions and answers using a clean, intuitive UI.
- Displays a default informational message if no flashcards are available.

### 📝 Quiz Mode with Objective Scoring
- Flashcards are converted into **multiple-choice quiz questions**.
- A **Submit Quiz** button calculates the final score only after completion.
- Final score is displayed as **X / Total Questions**.

### ⚠️ Robust Error Handling
- Empty study material input validation
- Flashcard count limit enforcement (max 20)
- Graceful handling of AI/API failures
- Safe handling when navigating Review or Quiz modes without generated flashcards
- No technical error messages exposed to the user

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – UI and application state management
- **CHATGPT API** – LLM-based flashcard generation
- **Cursor** – AI coding assistant used for development
- **python-dotenv** – Secure API key management

## 📌 Notes

- This application is designed to be **simple, reliable, and production-ready**.
- Emphasis is placed on usability, deterministic scoring, and clean user experience.
- The project meets coursework requirements for an AI-assisted, end-to-end functional application.

## Screenshots of App Features
### Main Navigation and Flashcard Generator Page
<img width="2758" height="1454" alt="image" src="https://github.com/user-attachments/assets/74562ef9-c5cc-4aa8-a3de-9dc20defb5e1" />

### Generated Flashcards
<img width="1995" height="952" alt="image" src="https://github.com/user-attachments/assets/45ea8b69-7e6b-4917-8e63-7f4574aa9d04" />

### Review Flashcards page
<img width="1995" height="1280" alt="image" src="https://github.com/user-attachments/assets/c95d0f85-d50a-4397-bc39-830a770b3208" />

### Quiz Mode Feature
<img width="2072" height="1305" alt="image" src="https://github.com/user-attachments/assets/07be81d9-919c-4c06-984c-c1b1b74aa9b3" />

