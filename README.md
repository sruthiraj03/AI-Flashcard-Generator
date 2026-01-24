# AI Study Flashcard Generator

A **Streamlit-based Study Flashcard Generator** built using **AI-assisted development with Cursor**.  
The application allows users to input study material, automatically generate flashcards, and test their knowledge through an interactive quiz with scoring.

This project demonstrates effective use of **natural language prompts to an AI coding assistant** to build a functional application end-to-end.

---

## 🚀 Features

- 📄 **Input Study Material**  
  Paste any study text (notes, definitions, concepts).

- 🧠 **AI-Generated Flashcards**  
  Flashcards are generated automatically using an LLM (with graceful fallback if API quota is exceeded).

- 👀 **Flashcard Review Mode**  
  View questions and reveal answers using a simple, clean UI.

- 📝 **Quiz Mode with Scoring**  
  Test your understanding by answering flashcard questions and receive a final score.

- ⚠️ **Robust Error Handling**  
  - Empty input handling  
  - Missing API key detection  
  - OpenAI quota error fallback  
  - No broken features

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** (UI)
- **OpenAI API** (LLM-based flashcard generation)
- **Cursor** (AI coding assistant)
- **python-dotenv** (secure API key management)

