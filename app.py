"""
Study Flashcard Generator - A Streamlit Application

This application helps students create and study flashcards from their study material.
It uses AI to generate flashcards and provides quiz functionality with automatic scoring.

Features:
- Generate flashcards from study material using AI
- Review flashcards with show/hide answer functionality
- Take quizzes with automatic answer scoring
"""

import streamlit as st
import json
import os
import time
import random
from difflib import SequenceMatcher
from openai import OpenAI
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

# Load API key from .env file (if it exists)
# This allows users to store their API key in a .env file instead of environment variables
load_dotenv()

# Get API key from environment variable
# The API key is required for generating flashcards using OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if API key is set - show friendly error if missing
# Note: We don't stop the app here, but will show error when user tries to generate
if not OPENAI_API_KEY:
    st.warning("⚠️ OpenAI API key not found. You'll need to set OPENAI_API_KEY to generate flashcards.")

# Configure the Streamlit page
# This sets the title, icon, and layout that appears in the browser tab
st.set_page_config(
    page_title="Study Flashcard Generator",
    page_icon="📚",
    layout="wide"  # Use wide layout for better use of screen space
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
# Session state stores data that persists across user interactions
# We initialize all variables with safe default values to prevent crashes

# Store all generated flashcards (list of dictionaries with 'question' and 'answer')
if 'flashcards' not in st.session_state:
    st.session_state.flashcards = []

# Track whether user is currently taking a quiz
if 'quiz_mode' not in st.session_state:
    st.session_state.quiz_mode = False

# Store the subset of flashcards selected for the current quiz
if 'quiz_flashcards' not in st.session_state:
    st.session_state.quiz_flashcards = []

# Track which question the user is currently on in the quiz
if 'current_quiz_index' not in st.session_state:
    st.session_state.current_quiz_index = 0

# Store user's answers for each quiz question (dictionary: index -> answer text)
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {}

# Store scores for each answer (dictionary: index -> score info)
if 'quiz_scores' not in st.session_state:
    st.session_state.quiz_scores = {}

# Store multiple-choice questions for the quiz (dictionary: index -> mcq data)
if 'quiz_mcq_questions' not in st.session_state:
    st.session_state.quiz_mcq_questions = {}

# Track whether the quiz has been completed
if 'quiz_completed' not in st.session_state:
    st.session_state.quiz_completed = False

# Track whether the quiz has been submitted (for scoring)
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False

# Legacy state variable (kept for compatibility)
if 'show_answer' not in st.session_state:
    st.session_state.show_answer = False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_flashcards(text, num_flashcards=5, max_retries=3):
    """
    Generate flashcards from study material using OpenAI's LLM.
    
    This function sends the study material to an AI model and asks it to create
    question-answer pairs. It includes retry logic in case the AI returns invalid JSON.
    
    Args:
        text (str): The study material text to generate flashcards from
        num_flashcards (int): Number of flashcards to generate (default: 5, max: 20)
        max_retries (int): Maximum number of retry attempts if JSON parsing fails (default: 3)
    
    Returns:
        list: List of dictionaries, each with 'question' and 'answer' keys
    
    Raises:
        ValueError: If API key is missing or if all retry attempts fail
    """
    # Validate inputs to prevent errors
    if not text or not isinstance(text, str):
        raise ValueError("Study material text is required and must be a string.")
    
    if not isinstance(num_flashcards, int) or num_flashcards < 1 or num_flashcards > 20:
        raise ValueError("Number of flashcards must be between 1 and 20.")
    
    # Check for API key - provide helpful error message
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it in your environment or .env file."
        )
    
    # Initialize OpenAI client with error handling
    # If initialization fails, it's likely a configuration issue, not an API key issue
    # (API key issues usually show up during API calls, not initialization)
    try:
        client = OpenAI(api_key=api_key)
    except Exception:
        # If client initialization fails, treat it as a generation failure
        # This will show a user-friendly message instead of technical details
        raise Exception("GENERATION_FAILED")
    
    # Create a detailed prompt that instructs the AI on how to create flashcards
    # The prompt emphasizes quality: no trivial questions, focus on key concepts
    prompt = f"""You are an educational assistant that creates high-quality study flashcards from text material.

Study Material:
{text}

Create exactly {num_flashcards} flashcards from the study material above. Each flashcard should have a clear question and a comprehensive answer.

CRITICAL GUIDELINES FOR QUESTIONS:
- AVOID trivial questions, yes/no questions, or questions that can be answered with a single word
- FOCUS on definitions, explanations, key concepts, relationships, processes, and applications
- Create questions that require understanding and explanation, not just recall
- Use question formats like: "What is...", "Explain...", "How does...", "Describe...", "What are the key components of...", "What is the relationship between..."
- Questions should test comprehension and understanding of important concepts

CRITICAL GUIDELINES FOR ANSWERS:
- Answers must be CONCISE but COMPLETE - provide enough detail to fully answer the question
- Aim for 2-4 sentences that capture the essential information
- Include key details, definitions, or explanations that demonstrate understanding
- Avoid overly verbose answers, but ensure completeness
- Focus on accuracy and clarity

IMPORTANT: Return ONLY valid JSON in this exact format (no markdown, no code blocks, no additional text):
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]

Requirements:
- Return exactly {num_flashcards} flashcards
- Each question must test understanding of key concepts (no trivial or yes/no questions)
- Each answer must be concise but complete (2-4 sentences)
- Return ONLY the JSON array, nothing else"""
    
    # Retry logic: if the AI returns invalid JSON, we try again
    # This handles cases where the AI might include markdown or extra text
    for attempt in range(max_retries):
        try:
            # Call OpenAI API to generate flashcards
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Using GPT-3.5 for cost-effectiveness
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that returns only valid JSON. Never include markdown code blocks or additional text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,  # Balance between creativity and consistency
                max_tokens=2000  # Enough tokens for multiple flashcards
            )
            
            # Extract the response text from the API response
            # Use safe access with .get() to prevent crashes if response structure is unexpected
            if not response or not response.choices or len(response.choices) == 0:
                raise Exception("GENERATION_FAILED")
            
            response_text = response.choices[0].message.content
            if not response_text:
                raise Exception("GENERATION_FAILED")
            
            response_text = response_text.strip()
            
            # Remove markdown code blocks if the AI included them (common issue)
            # This makes the function more robust
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            elif response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove closing ```
            response_text = response_text.strip()
            
            # Parse the JSON response
            # This will raise JSONDecodeError if the response isn't valid JSON
            flashcards = json.loads(response_text)
            
            # Validate that we got a list (not a dictionary or other type)
            if not isinstance(flashcards, list):
                raise Exception("GENERATION_FAILED")
            
            # Validate each flashcard has the required structure
            # This prevents crashes if the AI returns malformed data
            # If validation fails, treat it as a generation failure (show friendly error)
            validated_flashcards = []
            for i, card in enumerate(flashcards):
                # Validate structure - if any check fails, raise friendly error
                if not isinstance(card, dict):
                    raise Exception("GENERATION_FAILED")
                
                # Check for required keys
                if "question" not in card or "answer" not in card:
                    raise Exception("GENERATION_FAILED")
                
                # Ensure question and answer are strings
                if not isinstance(card.get("question"), str) or not isinstance(card.get("answer"), str):
                    raise Exception("GENERATION_FAILED")
                
                # Ensure question and answer are not empty
                if not card.get("question", "").strip() or not card.get("answer", "").strip():
                    raise Exception("GENERATION_FAILED")
                
                # If all validations pass, add to validated list
                validated_flashcards.append({
                    "question": card["question"].strip(),
                    "answer": card["answer"].strip()
                })
            
            # Return the validated flashcards (limit to requested number)
            return validated_flashcards[:num_flashcards]
            
        except json.JSONDecodeError:
            # If JSON parsing fails, wait and retry (exponential backoff)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Wait 1s, 2s, 4s...
                time.sleep(wait_time)
                continue
            else:
                # All retries failed - raise a user-friendly exception
                raise Exception("GENERATION_FAILED")
        
        except Exception as e:
            # Handle any other unexpected errors (API errors, rate limits, connection issues, etc.)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                # For any error (rate limits, API errors, connection issues, etc.), 
                # raise a generic user-friendly exception
                raise Exception("GENERATION_FAILED")
    
    # Should not reach here, but included as a safety net
    raise Exception("GENERATION_FAILED")


def calculate_answer_similarity(user_answer, correct_answer):
    """
    Calculate how similar a user's answer is to the correct answer.
    
    This function uses multiple methods to score answers:
    1. Character-level similarity (how similar the text is)
    2. Keyword matching (how many important words match)
    3. Phrase matching (how many key phrases appear)
    
    Args:
        user_answer (str): The answer the user typed
        correct_answer (str): The correct answer from the flashcard
    
    Returns:
        tuple: (similarity_score, is_correct, matched_keywords)
            - similarity_score (float): Score between 0.0 and 1.0
            - is_correct (bool): True if similarity >= 0.4 (40%)
            - matched_keywords (list): List of keywords that matched
    """
    # Handle empty or invalid inputs safely
    if not user_answer or not isinstance(user_answer, str):
        return 0.0, False, []
    
    if not correct_answer or not isinstance(correct_answer, str):
        # If correct answer is invalid, we can't score - return neutral score
        return 0.5, False, []
    
    # Normalize answers: convert to lowercase and remove extra whitespace
    # This makes comparison more fair (ignores case and spacing differences)
    user_norm = ' '.join(user_answer.lower().split())
    correct_norm = ' '.join(correct_answer.lower().split())
    
    # Method 1: Sequence similarity (character-level comparison)
    # This compares how similar the actual text is character by character
    try:
        sequence_similarity = SequenceMatcher(None, user_norm, correct_norm).ratio()
    except Exception:
        # If comparison fails, default to 0
        sequence_similarity = 0.0
    
    # Method 2: Keyword matching
    # Extract important words (excluding common stop words like "the", "a", etc.)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'
    }
    
    # Get important words from both answers (words longer than 2 characters, not stop words)
    correct_words = set(
        word for word in correct_norm.split()
        if word not in stop_words and len(word) > 2
    )
    user_words = set(
        word for word in user_norm.split()
        if word not in stop_words and len(word) > 2
    )
    
    # Calculate what percentage of important words from correct answer appear in user answer
    if correct_words:
        keyword_match_ratio = len(user_words.intersection(correct_words)) / len(correct_words)
    else:
        keyword_match_ratio = 0.0
    
    # Track which keywords matched (useful for feedback)
    matched_keywords = list(user_words.intersection(correct_words))
    
    # Method 3: Phrase matching
    # Check if key phrases from the correct answer appear in the user's answer
    # This catches cases where the user explains the concept in their own words
    try:
        correct_phrases = [phrase.strip() for phrase in correct_norm.split('.') if phrase.strip()]
        phrase_match_count = sum(
            1 for phrase in correct_phrases
            if phrase in user_norm and len(phrase) > 10
        )
        phrase_match_ratio = phrase_match_count / len(correct_phrases) if correct_phrases else 0.0
    except Exception:
        phrase_match_ratio = 0.0
    
    # Combine all three methods with weighted average
    # Sequence similarity: 40%, Keyword matching: 40%, Phrase matching: 20%
    combined_similarity = (
        0.4 * sequence_similarity +
        0.4 * keyword_match_ratio +
        0.2 * phrase_match_ratio
    )
    
    # Ensure similarity is between 0 and 1
    combined_similarity = max(0.0, min(1.0, combined_similarity))
    
    # Consider answer correct if similarity is above 40% threshold
    # This threshold balances being fair while still requiring some accuracy
    is_correct = combined_similarity >= 0.4
    
    return combined_similarity, is_correct, matched_keywords


def generate_multiple_choice_questions(flashcards, max_retries=3):
    """
    Convert flashcards into multiple-choice questions with 4 options each.
    
    This function uses an LLM to convert Q&A flashcards into multiple-choice format
    with exactly 4 options (A, B, C, D) where only one is correct.
    
    Args:
        flashcards (list): List of flashcard dictionaries with 'question' and 'answer' keys
        max_retries (int): Maximum number of retry attempts if JSON parsing fails (default: 3)
    
    Returns:
        dict: Dictionary mapping index to MCQ data with structure:
            {
                'question': str,
                'options': {'A': str, 'B': str, 'C': str, 'D': str},
                'correct_answer': str (one of 'A', 'B', 'C', 'D')
            }
    
    Raises:
        Exception: If generation fails after all retries
    """
    # Validate inputs
    if not flashcards or not isinstance(flashcards, list) or len(flashcards) == 0:
        raise Exception("GENERATION_FAILED")
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    
    try:
        client = OpenAI(api_key=api_key)
    except Exception:
        raise Exception("GENERATION_FAILED")
    
    # Prepare flashcards data for the prompt
    flashcards_text = ""
    for i, card in enumerate(flashcards):
        flashcards_text += f"Flashcard {i+1}:\n"
        flashcards_text += f"Question: {card.get('question', '')}\n"
        flashcards_text += f"Answer: {card.get('answer', '')}\n\n"
    
    # Create prompt for multiple-choice question generation
    prompt = f"""You are an educational assistant that converts flashcards into multiple-choice questions.

Flashcards to convert:
{flashcards_text}

For each flashcard, create a multiple-choice question with exactly 4 options (A, B, C, D).

REQUIREMENTS:
- Convert the question into a clear multiple-choice format
- Provide exactly 4 answer options labeled A, B, C, D
- Only ONE option must be correct (the correct answer from the flashcard)
- The other 3 options must be plausible but clearly wrong distractors
- Distractors should be related to the topic but incorrect
- Clearly indicate which option (A, B, C, or D) is the correct answer

IMPORTANT: Return ONLY valid JSON in this exact format (no markdown, no code blocks, no additional text):
[
  {{
    "question": "The converted multiple-choice question text",
    "options": {{
      "A": "First option text",
      "B": "Second option text",
      "C": "Third option text",
      "D": "Fourth option text"
    }},
    "correct_answer": "A"
  }},
  {{
    "question": "...",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "correct_answer": "B"
  }}
]

Requirements:
- Return exactly {len(flashcards)} multiple-choice questions (one per flashcard)
- Each question must have exactly 4 options (A, B, C, D)
- correct_answer must be exactly one of: "A", "B", "C", or "D"
- Return ONLY the JSON array, nothing else"""
    
    # Retry logic for JSON parsing
    for attempt in range(max_retries):
        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that returns only valid JSON. Never include markdown code blocks or additional text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=3000
            )
            
            # Extract response
            if not response or not response.choices or len(response.choices) == 0:
                raise Exception("GENERATION_FAILED")
            
            response_text = response.choices[0].message.content
            if not response_text:
                raise Exception("GENERATION_FAILED")
            
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            elif response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            mcq_list = json.loads(response_text)
            
            # Validate structure
            if not isinstance(mcq_list, list):
                raise Exception("GENERATION_FAILED")
            
            if len(mcq_list) != len(flashcards):
                raise Exception("GENERATION_FAILED")
            
            # Validate and convert to dictionary format
            mcq_dict = {}
            for idx, mcq in enumerate(mcq_list):
                try:
                    # Validate required fields
                    if not isinstance(mcq, dict):
                        raise Exception("GENERATION_FAILED")
                    
                    if "question" not in mcq or "options" not in mcq or "correct_answer" not in mcq:
                        raise Exception("GENERATION_FAILED")
                    
                    # Validate options structure
                    options = mcq["options"]
                    if not isinstance(options, dict):
                        raise Exception("GENERATION_FAILED")
                    
                    # Ensure all 4 options exist
                    required_options = ["A", "B", "C", "D"]
                    if not all(opt in options for opt in required_options):
                        raise Exception("GENERATION_FAILED")
                    
                    # Validate correct_answer
                    correct = mcq["correct_answer"]
                    if correct not in required_options:
                        raise Exception("GENERATION_FAILED")
                    
                    # Validate all are strings
                    if not isinstance(mcq["question"], str) or not all(isinstance(options[opt], str) for opt in required_options):
                        raise Exception("GENERATION_FAILED")
                    
                    # Store in dictionary
                    mcq_dict[idx] = {
                        "question": mcq["question"].strip(),
                        "options": {
                            "A": options["A"].strip(),
                            "B": options["B"].strip(),
                            "C": options["C"].strip(),
                            "D": options["D"].strip()
                        },
                        "correct_answer": correct
                    }
                except Exception:
                    raise Exception("GENERATION_FAILED")
            
            return mcq_dict
            
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                raise Exception("GENERATION_FAILED")
        
        except Exception:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                raise Exception("GENERATION_FAILED")
    
    raise Exception("GENERATION_FAILED")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application function that handles the UI and user interactions.
    
    This function creates the Streamlit 🗂️interface with three main sections:
    1. Generate Flashcards - Create new flashcards from study material
    2. Review Flashcards - Review existing flashcards
    3. Quiz Mode - Take a quiz on flashcards
    """
    try:
        # Display main header
        st.header("🗂️ Study Flashcard Generator")
        
        # Create sidebar navigation
        # Users can switch between different modes using the radio buttons
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Choose a mode:",
            ["Generate Flashcards", "Review Flashcards", "Quiz Mode"],
            help="Select which section of the app you want to use"
        )
        
        # ====================================================================
        # GENERATE FLASHCARDS SECTION
        # ====================================================================
        if page == "Generate Flashcards":
            st.header("📝 Generate New Flashcards")
            st.markdown(
                "Paste your study material below and click 'Generate Flashcards'"
            )
            
            # Input for number of flashcards to generate
            # Validate input range to prevent errors
            try:
                num_flashcards = st.number_input(
                    "Number of Flashcards",
                    min_value=1,
                    max_value=20,
                    value=5,
                    help="Select how many flashcards you want to generate (1-20)"
                )
            except Exception as e:
                st.error(f"Error with number input: {str(e)}")
                num_flashcards = 5  # Safe default
            
            # Text area for study material input
            study_material = st.text_area(
                "Study Material",
                height=300,
                placeholder=(
                    "Paste your study material here...\n\n"
                    "Example:\n"
                    "Machine learning is a subset of artificial intelligence that enables "
                    "systems to learn and improve from experience without being explicitly programmed."
                )
            )
            
            # Generate button (centered)
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                generate_button = st.button(
                    "🔄 Generate Flashcards",
                    type="primary",
                    use_container_width=True
                )
            
            # Handle flashcard generation when button is clicked
            if generate_button:
                # Validate input before processing
                if not study_material or not study_material.strip():
                    st.error("⚠️ Please enter some study material before generating flashcards.")
                elif len(study_material.strip()) < 20:
                    st.warning("⚠️ Study material seems too short. Please provide more content for better flashcards.")
                else:
                    try:
                        # Show loading spinner while generating
                        with st.spinner("Generating flashcards with AI... This may take a few seconds."):
                            # Generate flashcards using the helper function
                            new_flashcards = generate_flashcards(
                                study_material,
                                num_flashcards=num_flashcards
                            )
                            
                            # Validate that we got flashcards back
                            if not new_flashcards or len(new_flashcards) == 0:
                                st.error("⚠️ No flashcards were generated. Please try again.")
                            else:
                                # Store flashcards in session state
                                st.session_state.flashcards = new_flashcards
                                
                                # RESET ALL QUIZ-RELATED STATE when new flashcards are generated
                                # This ensures Quiz Mode starts fresh with the new flashcards
                                st.session_state.quiz_mode = False
                                st.session_state.quiz_flashcards = []  # Clear selected quiz flashcards
                                st.session_state.quiz_mcq_questions = {}  # Clear generated MCQ questions
                                st.session_state.current_quiz_index = 0  # Reset to first question
                                st.session_state.quiz_answers = {}  # Clear all user answers
                                st.session_state.quiz_scores = {}  # Clear all scores
                                st.session_state.quiz_completed = False  # Reset completion status
                                st.session_state.quiz_submitted = False  # Reset submission status
                                
                                # Clear any per-question answer selection state
                                # Remove all keys that start with "selected_answer_" or "option_"
                                keys_to_remove = [
                                    key for key in st.session_state.keys()
                                    if key.startswith("selected_answer_") or 
                                       key.startswith("option_") or
                                       key.startswith("radio_") or
                                       key.startswith("select_")
                                ]
                                for key in keys_to_remove:
                                    del st.session_state[key]
                                
                                st.success(
                                    f"✅ Successfully generated {len(new_flashcards)} flashcards!"
                                )
                                
                                # Display preview of generated flashcards
                                st.subheader("📋 Generated Flashcards Preview")
                                for i, card in enumerate(new_flashcards, 1):
                                    # Safely access card data with error handling
                                    try:
                                        question = card.get('question', 'No question')[:50]
                                        with st.expander(f"Flashcard {i}: {question}..."):
                                            st.write("**Question:**", card.get('question', 'N/A'))
                                            st.write("**Answer:**", card.get('answer', 'N/A'))
                                    except Exception as e:
                                        st.error(f"Error displaying flashcard {i}: {str(e)}")
                    
                    except ValueError as e:
                        # Handle API key missing - this is the only technical error we show
                        error_message = str(e)
                        if "OPENAI_API_KEY" in error_message or "API key" in error_message:
                            st.error("""
                            🔑 **API Key Missing or Invalid**
                            
                            Please set your OpenAI API key as an environment variable:
                            
                            **Windows (PowerShell):**
                            ```powershell
                            $env:OPENAI_API_KEY="your-api-key-here"
                            ```
                            
                            **Windows (Command Prompt):**
                            ```cmd
                            set OPENAI_API_KEY=your-api-key-here
                            ```
                            
                            **Linux/Mac:**
                            ```bash
                            export OPENAI_API_KEY="your-api-key-here"
                            ```
                            
                            Or create a `.env` file in your project directory with:
                            ```
                            OPENAI_API_KEY=your-api-key-here
                            ```
                            """)
                        else:
                            # For other ValueError cases, show user-friendly message
                            st.info("""
                            ⚠️ **Unable to Generate Flashcards**
                            
                            We're unable to generate flashcards at the moment. This might be due to:
                            - The service is temporarily unavailable
                            - Your input needs adjustment
                            
                            **What you can try:**
                            - Wait a moment and try again
                            - Edit your study material (try making it shorter or more focused)
                            - Check that your study material contains enough content
                            """)
                    
                    except Exception as e:
                        # Handle all other errors (API errors, rate limits, connection issues, etc.)
                        # Show a calm, user-friendly message without technical details
                        if str(e) == "GENERATION_FAILED":
                            # This is our custom exception for user-friendly errors
                            st.info("""
                            ⚠️ **Unable to Generate Flashcards**
                            
                            We're unable to generate flashcards at the moment. This might be temporary.
                            
                            **What you can try:**
                            - Wait a few moments and try again
                            - Edit your study material (try making it shorter or more focused)
                            - Ensure your study material contains clear, complete information
                            """)
                        else:
                            # Catch any other unexpected exceptions with the same friendly message
                            st.info("""
                            ⚠️ **Unable to Generate Flashcards**
                            
                            We're unable to generate flashcards at the moment. This might be temporary.
                            
                            **What you can try:**
                            - Wait a few moments and try again
                            - Edit your study material (try making it shorter or more focused)
                            - Ensure your study material contains clear, complete information
                            """)
            
            # Display current flashcards count (if any exist)
            if st.session_state.flashcards:
                try:
                    count = len(st.session_state.flashcards)
                    st.info(f"💾 You currently have {count} flashcards stored.")
                except Exception:
                    pass  # Silently handle if count fails
        
        # ====================================================================
        # REVIEW FLASHCARDS SECTION
        # ====================================================================
        elif page == "Review Flashcards":
            st.header("🔍 Review Flashcards")
            
            # Check if flashcards exist before trying to display them
            if not st.session_state.flashcards or len(st.session_state.flashcards) == 0:
                st.warning(
                    "⚠️ No flashcards available. Please generate flashcards first "
                    "in the 'Generate Flashcards' section."
                )
            else:
                try:
                    flashcard_count = len(st.session_state.flashcards)
                    st.markdown(
                        f"Review your {flashcard_count} flashcards. "
                        "Click on each card to expand and reveal the answer."
                    )
                    
                    # Display each flashcard in an expandable section
                    for idx, card in enumerate(st.session_state.flashcards):
                        try:
                            # Safely get card data with defaults
                            question = card.get('question', 'No question available')
                            answer = card.get('answer', 'No answer available')
                            
                            # Initialize answer visibility state for this card
                            answer_key = f"show_answer_{idx}"
                            if answer_key not in st.session_state:
                                st.session_state[answer_key] = False
                            
                            # Create expandable card with question as header
                            with st.expander(
                                f"📌 **Card {idx + 1}:** {question}",
                                expanded=False
                            ):
                                # Display question
                                st.markdown("**❓ Question:**")
                                st.info(question)
                                
                                st.markdown("")  # Spacing
                                
                                # Button to toggle answer visibility
                                button_text = (
                                    "🙈 Hide Answer" if st.session_state[answer_key]
                                    else "👁️ Reveal Answer"
                                )
                                if st.button(button_text, key=f"toggle_{idx}", use_container_width=True):
                                    st.session_state[answer_key] = not st.session_state[answer_key]
                                
                                st.markdown("")  # Spacing
                                
                                # Display answer if revealed
                                if st.session_state[answer_key]:
                                    st.markdown("**💡 Answer:**")
                                    st.success(answer)
                                else:
                                    st.markdown("_💡 Click 'Reveal Answer' above to see the solution_")
                                
                                st.markdown("---")
                        
                        except Exception as e:
                            # If one card fails, show error but continue with others
                            st.error(f"Error displaying flashcard {idx + 1}: {str(e)}")
                            continue
                
                except Exception as e:
                    st.error(f"Error loading flashcards: {str(e)}")
                    st.info("💡 Try generating new flashcards or refreshing the page.")
        
        # ====================================================================
        # QUIZ MODE SECTION
        # ====================================================================
        elif page == "Quiz Mode":
            st.header("🎯 Quiz Mode")
            
            # Check if flashcards exist before starting quiz
            if not st.session_state.flashcards or len(st.session_state.flashcards) == 0:
                st.warning(
                    "⚠️ No flashcards available. Please generate flashcards first "
                    "in the 'Generate Flashcards' section."
                )
            else:
                # Quiz setup phase (before quiz starts)
                if not st.session_state.quiz_mode:
                    st.markdown(
                        "Test your knowledge! Answer the questions and see your score at the end."
                    )
                    
                    try:
                        # Calculate how many flashcards are available
                        total_available = len(st.session_state.flashcards)
                        max_quiz_size = min(total_available, 20)  # Cap at 20 for performance
                        
                        # Let user select how many flashcards to quiz on
                        num_quiz_cards = st.number_input(
                            f"Number of flashcards to quiz on (out of {total_available} available)",
                            min_value=1,
                            max_value=max_quiz_size,
                            value=min(5, total_available),
                            help="Select how many flashcards you want to be quizzed on. They will be randomly selected."
                        )
                        
                        # Start quiz button
                        if st.button("🚀 Start Quiz", type="primary"):
                            try:
                                # Validate that we have flashcards to use
                                if not st.session_state.flashcards:
                                    st.error("No flashcards available. Please generate flashcards first.")
                                else:
                                    # Randomly select subset of flashcards
                                    all_flashcards = st.session_state.flashcards.copy()
                                    random.shuffle(all_flashcards)
                                    selected_flashcards = all_flashcards[:num_quiz_cards]
                                    
                                    # Validate selection
                                    if not selected_flashcards or len(selected_flashcards) == 0:
                                        st.error("Failed to select flashcards. Please try again.")
                                    else:
                                        # Store selected flashcards
                                        st.session_state.quiz_flashcards = selected_flashcards
                                        
                                        # Generate multiple-choice questions from flashcards
                                        with st.spinner("Generating multiple-choice questions... This may take a few seconds."):
                                            try:
                                                mcq_questions = generate_multiple_choice_questions(selected_flashcards)
                                                st.session_state.quiz_mcq_questions = mcq_questions
                                            except ValueError as e:
                                                # API key error - show setup instructions
                                                error_message = str(e)
                                                if "OPENAI_API_KEY" in error_message or "API key" in error_message:
                                                    st.error("""
                                                    🔑 **API Key Missing or Invalid**
                                                    
                                                    Please set your OpenAI API key to generate quiz questions.
                                                    """)
                                                    return
                                                else:
                                                    st.info("""
                                                    ⚠️ **Unable to Generate Quiz Questions**
                                                    
                                                    We're unable to generate quiz questions at the moment.
                                                    
                                                    **What you can try:**
                                                    - Wait a few moments and try again
                                                    - Check your API key configuration
                                                    """)
                                                    return
                                            except Exception:
                                                # Any other error - show friendly message
                                                st.info("""
                                                ⚠️ **Unable to Generate Quiz Questions**
                                                
                                                We're unable to generate quiz questions at the moment. This might be temporary.
                                                
                                                **What you can try:**
                                                - Wait a few moments and try again
                                                - Try with fewer flashcards
                                                """)
                                                return
                                        
                                        # Initialize quiz state
                                        st.session_state.quiz_mode = True
                                        st.session_state.current_quiz_index = 0
                                        st.session_state.quiz_answers = {}  # Will store selected option (A, B, C, or D)
                                        st.session_state.quiz_scores = {}  # Will store is_correct boolean
                                        st.session_state.quiz_completed = False
                                        st.session_state.quiz_submitted = False  # Track if quiz has been submitted
                                        st.rerun()
                            
                            except Exception as e:
                                st.error(f"Error starting quiz: {str(e)}")
                                st.info("💡 Please try again or refresh the page.")
                    
                    except Exception as e:
                        st.error(f"Error in quiz setup: {str(e)}")
                
                # Quiz in progress or completed
                else:
                    # Safety check: If flashcards have changed (new flashcards generated),
                    # automatically reset quiz state to prevent stale data
                    if (not st.session_state.quiz_flashcards or
                            len(st.session_state.quiz_flashcards) == 0 or
                            not st.session_state.quiz_mcq_questions or
                            len(st.session_state.quiz_mcq_questions) == 0):
                        # Quiz data is missing or invalid - reset quiz mode
                        st.warning("⚠️ Quiz data not found or flashcards have been updated. Please start a new quiz.")
                        # Auto-reset quiz state
                        st.session_state.quiz_mode = False
                        st.session_state.quiz_flashcards = []
                        st.session_state.quiz_mcq_questions = {}
                        st.session_state.current_quiz_index = 0
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_scores = {}
                        st.session_state.quiz_completed = False
                        st.session_state.quiz_submitted = False
                        st.rerun()
                    
                    # Quiz is active
                    else:
                        try:
                            total_flashcards = len(st.session_state.quiz_flashcards)
                            current_index = st.session_state.current_quiz_index
                            
                            # Ensure current_index is valid
                            if current_index < 0:
                                current_index = 0
                                st.session_state.current_quiz_index = 0
                            if current_index >= total_flashcards:
                                st.session_state.quiz_completed = True
                                st.rerun()
                            
                            # Quiz in progress (not completed)
                            if not st.session_state.quiz_completed and current_index < total_flashcards:
                                # Check if MCQ questions are available
                                if not st.session_state.quiz_mcq_questions or current_index not in st.session_state.quiz_mcq_questions:
                                    st.error("Quiz questions not available. Please start a new quiz.")
                                    if st.button("🔄 Reset Quiz"):
                                        st.session_state.quiz_mode = False
                                        st.session_state.quiz_completed = False
                                        st.session_state.quiz_submitted = False
                                        st.rerun()
                                    return
                                
                                # Progress bar
                                try:
                                    progress = (current_index + 1) / total_flashcards
                                    st.progress(progress)
                                    st.caption(f"Question {current_index + 1} of {total_flashcards}")
                                except Exception:
                                    pass  # Silently handle progress bar errors
                                
                                # Get current multiple-choice question
                                try:
                                    mcq = st.session_state.quiz_mcq_questions[current_index]
                                    question = mcq.get('question', 'No question available')
                                    options = mcq.get('options', {})
                                    correct_answer = mcq.get('correct_answer', '')
                                except (KeyError, TypeError) as e:
                                    st.error(f"Error loading question {current_index + 1}: {str(e)}")
                                    st.info("💡 Please reset the quiz and try again.")
                                    if st.button("🔄 Reset Quiz"):
                                        st.session_state.quiz_mode = False
                                        st.session_state.quiz_completed = False
                                        st.session_state.quiz_submitted = False
                                        st.rerun()
                                    return
                                
                                # Display current question
                                st.markdown("### Question:")
                                st.info(question)
                                
                                # Initialize answer selection state - start with None (no selection)
                                answer_key = f"selected_answer_{current_index}"
                                if answer_key not in st.session_state:
                                    st.session_state[answer_key] = None
                                
                                # Get previously selected answer (if any) - this preserves selection when navigating
                                previous_selection = st.session_state.get(answer_key)
                                
                                # Display multiple-choice options as clickable buttons
                                # This gives us full control - no option is pre-selected
                                st.markdown("**Select your answer:**")
                                
                                # Create buttons for each option in a clean layout
                                option_cols = st.columns(4)
                                selected_option = None
                                
                                # Display each option as a button
                                # Disable buttons if quiz has been submitted
                                for idx, opt in enumerate(["A", "B", "C", "D"]):
                                    with option_cols[idx]:
                                        # Highlight button if this was previously selected
                                        button_type = "primary" if previous_selection == opt else "secondary"
                                        button_label = f"**{opt}**\n\n{options.get(opt, '')}"
                                        
                                        if st.button(
                                            button_label,
                                            key=f"option_{current_index}_{opt}",
                                            use_container_width=True,
                                            type=button_type,
                                            disabled=st.session_state.quiz_submitted  # Lock after submission
                                        ):
                                            # User clicked this option
                                            selected_option = opt
                                
                                # Process selection only if user clicked a button and quiz is not submitted
                                if selected_option is not None and not st.session_state.quiz_submitted:
                                    # Update state when user actively selects an option
                                    if selected_option != st.session_state.get(answer_key):
                                        st.session_state[answer_key] = selected_option
                                        st.session_state.quiz_answers[current_index] = selected_option
                                        st.rerun()  # Rerun to update UI
                                
                                # Show status message
                                if st.session_state.quiz_submitted:
                                    # Quiz is submitted - show if answer was correct/incorrect
                                    if previous_selection is not None:
                                        is_correct = (previous_selection == correct_answer)
                                        if is_correct:
                                            st.success("✅ Correct!")
                                        else:
                                            st.error(f"❌ Incorrect. The correct answer is {correct_answer}.")
                                    else:
                                        st.warning("⚠️ This question was not answered.")
                                elif previous_selection is None:
                                    # No answer selected yet (quiz not submitted)
                                    st.info("👆 Click one of the options above to select your answer.")
                                else:
                                    # Answer selected but quiz not submitted yet
                                    st.info(f"Selected: **{previous_selection}** (Click 'Submit Quiz' when done to see results)")
                                
                                # Navigation buttons
                                col1, col2, col3 = st.columns([1, 1, 1])
                                
                                # Previous button
                                with col1:
                                    if st.button("⏮️ Previous", disabled=(current_index == 0 or st.session_state.quiz_submitted)):
                                        if current_index > 0:
                                            st.session_state.current_quiz_index -= 1
                                            st.rerun()
                                
                                # Next button or Submit Quiz button
                                with col2:
                                    # Show "Submit Quiz" on last question, "Next" otherwise
                                    if current_index == total_flashcards - 1:
                                        # Last question - show Submit Quiz button
                                        if st.button("✅ Submit Quiz", type="primary", use_container_width=True, disabled=st.session_state.quiz_submitted):
                                            # Calculate and store scores
                                            st.session_state.quiz_submitted = True
                                            
                                            # Score all answers
                                            for idx in range(total_flashcards):
                                                try:
                                                    selected_option = st.session_state.quiz_answers.get(idx, None)
                                                    
                                                    if idx not in st.session_state.quiz_mcq_questions:
                                                        continue
                                                    
                                                    mcq = st.session_state.quiz_mcq_questions[idx]
                                                    correct_option = mcq.get('correct_answer', '')
                                                    
                                                    # Score: 1 point if correct, 0 if incorrect or unanswered
                                                    is_correct = (selected_option == correct_option) if selected_option is not None else False
                                                    st.session_state.quiz_scores[idx] = {
                                                        'is_correct': is_correct
                                                    }
                                                except Exception:
                                                    continue
                                            
                                            st.session_state.quiz_completed = True
                                            st.rerun()
                                    else:
                                        # Not last question - show Next button
                                        if st.button("⏭️ Next", use_container_width=True, disabled=st.session_state.quiz_submitted):
                                            # Move to next question
                                            st.session_state.current_quiz_index += 1
                                            st.rerun()
                                
                                # Show correct answer only after submission
                                with col3:
                                    if st.session_state.quiz_submitted:
                                        st.caption(f"Correct: **{correct_answer}**")
                                    else:
                                        st.caption("Answer all questions, then click 'Submit Quiz'")
                            
                            # Quiz completed - show results after submission
                            elif st.session_state.quiz_completed and st.session_state.quiz_submitted:
                                st.balloons()  # Celebration animation
                                st.success("🎉 Quiz Submitted!")
                                
                                try:
                                    # Calculate final score from stored scores
                                    total = len(st.session_state.quiz_flashcards)
                                    correct_count = 0
                                    incorrect_answers = []
                                    
                                    # Count correct/incorrect from stored scores
                                    for idx in range(total):
                                        try:
                                            score_info = st.session_state.quiz_scores.get(idx, {})
                                            if score_info.get('is_correct', False):
                                                correct_count += 1
                                            else:
                                                # Get question data for review
                                                if idx in st.session_state.quiz_mcq_questions:
                                                    mcq = st.session_state.quiz_mcq_questions[idx]
                                                    selected_option = st.session_state.quiz_answers.get(idx, None)
                                                    incorrect_answers.append({
                                                        'index': idx,
                                                        'mcq': mcq,
                                                        'selected_option': selected_option
                                                    })
                                        except Exception:
                                            continue
                                    
                                    # Display final score - simple format: "Final Score: X / N"
                                    st.markdown("---")
                                    percentage = (correct_count / total * 100) if total > 0 else 0
                                    st.markdown(f"## Final Score: **{correct_count} / {total}** ({percentage:.1f}%)")
                                    st.markdown("---")
                                    
                                    # Show review of incorrect answers
                                    if incorrect_answers:
                                        st.subheader("❌ Review Incorrect Answers")
                                        st.markdown(
                                            "Here are the questions you got wrong. "
                                            "Review them to improve your understanding."
                                        )
                                        
                                        for item in incorrect_answers:
                                            try:
                                                idx = item.get('index', 0)
                                                mcq = item.get('mcq', {})
                                                selected_option = item.get('selected_option', None)
                                                
                                                question = mcq.get('question', 'No question')[:60]
                                                options = mcq.get('options', {})
                                                correct_option = mcq.get('correct_answer', '')
                                                
                                                with st.expander(
                                                    f"❌ Question {idx + 1}: {question}...",
                                                    expanded=True
                                                ):
                                                    st.markdown("**❓ Question:**")
                                                    st.info(mcq.get('question', 'N/A'))
                                                    
                                                    st.markdown("**📋 Options:**")
                                                    for opt in ["A", "B", "C", "D"]:
                                                        option_text = options.get(opt, '')
                                                        if opt == correct_option:
                                                            st.success(f"**{opt}.** {option_text} ✅ (Correct)")
                                                        elif opt == selected_option:
                                                            st.error(f"**{opt}.** {option_text} ❌ (Your Answer)")
                                                        else:
                                                            st.write(f"**{opt}.** {option_text}")
                                                    
                                                    # Show message if question was unanswered
                                                    if selected_option is None:
                                                        st.warning("⚠️ This question was not answered.")
                                            except Exception:
                                                continue
                                    else:
                                        st.success("🎊 Perfect! You got all answers correct!")
                                    
                                    # Show all answers summary
                                    st.subheader("📋 Complete Answer Review")
                                    for idx in range(total):
                                        try:
                                            if idx not in st.session_state.quiz_mcq_questions:
                                                continue
                                            
                                            mcq = st.session_state.quiz_mcq_questions[idx]
                                            selected_option = st.session_state.quiz_answers.get(idx, None)
                                            score_info = st.session_state.quiz_scores.get(idx, {})
                                            is_correct = score_info.get('is_correct', False)
                                            
                                            status_icon = "✅" if is_correct else "❌"
                                            question = mcq.get('question', 'No question')[:50]
                                            options = mcq.get('options', {})
                                            correct_option = mcq.get('correct_answer', '')
                                            
                                            with st.expander(
                                                f"{status_icon} Question {idx + 1}: {question}..."
                                            ):
                                                st.markdown("**Question:**")
                                                st.write(mcq.get('question', 'N/A'))
                                                
                                                st.markdown("**Options:**")
                                                for opt in ["A", "B", "C", "D"]:
                                                    option_text = options.get(opt, '')
                                                    if opt == correct_option:
                                                        st.success(f"**{opt}.** {option_text} ✅ (Correct Answer)")
                                                    elif opt == selected_option:
                                                        st.warning(f"**{opt}.** {option_text} (Your Answer)")
                                                    else:
                                                        st.write(f"**{opt}.** {option_text}")
                                                
                                                # Show message if question was unanswered
                                                if selected_option is None:
                                                    st.warning("⚠️ This question was not answered.")
                                        except Exception:
                                            continue
                                    
                                    # Reset quiz button
                                    col1, col2, col3 = st.columns([1, 1, 1])
                                    with col2:
                                        if st.button("🔄 Take Quiz Again", type="primary", use_container_width=True):
                                            try:
                                                st.session_state.quiz_mode = False
                                                st.session_state.quiz_flashcards = []
                                                st.session_state.quiz_mcq_questions = {}
                                                st.session_state.current_quiz_index = 0
                                                st.session_state.quiz_answers = {}
                                                st.session_state.quiz_scores = {}
                                                st.session_state.quiz_completed = False
                                                st.session_state.quiz_submitted = False
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Error resetting quiz: {str(e)}")
                                
                                except Exception as e:
                                    st.error(f"Error displaying quiz results: {str(e)}")
                                    st.info("💡 Please try resetting the quiz or refreshing the page.")
                        
                        except Exception as e:
                            st.error(f"Error in quiz mode: {str(e)}")
                            st.info("💡 Please try resetting the quiz or refreshing the page.")
    
    except Exception as e:
        # Catch-all error handler for the entire app
        st.error("⚠️ An unexpected error occurred in the application.")
        st.error(f"Error details: {str(e)}")
        st.info("💡 Please try refreshing the page. If the problem persists, check your inputs and try again.")


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the main application
    # This is the entry point when the script is executed
    main()
