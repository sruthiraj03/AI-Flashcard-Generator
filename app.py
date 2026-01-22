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

# Track whether the quiz has been completed
if 'quiz_completed' not in st.session_state:
    st.session_state.quiz_completed = False

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
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")
    
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
                raise ValueError("Empty response from OpenAI API")
            
            response_text = response.choices[0].message.content
            if not response_text:
                raise ValueError("No content in API response")
            
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
                raise ValueError("API response is not a list. Expected a list of flashcards.")
            
            # Validate each flashcard has the required structure
            # This prevents crashes if the AI returns malformed data
            validated_flashcards = []
            for i, card in enumerate(flashcards):
                if not isinstance(card, dict):
                    raise ValueError(f"Flashcard {i+1} is not a dictionary. Expected a dictionary with 'question' and 'answer' keys.")
                
                # Check for required keys
                if "question" not in card:
                    raise ValueError(f"Flashcard {i+1} is missing 'question' key.")
                if "answer" not in card:
                    raise ValueError(f"Flashcard {i+1} is missing 'answer' key.")
                
                # Ensure question and answer are strings
                if not isinstance(card.get("question"), str) or not isinstance(card.get("answer"), str):
                    raise ValueError(f"Flashcard {i+1} has invalid question or answer type. Both must be strings.")
                
                # Ensure question and answer are not empty
                if not card.get("question", "").strip() or not card.get("answer", "").strip():
                    raise ValueError(f"Flashcard {i+1} has empty question or answer.")
                
                validated_flashcards.append({
                    "question": card["question"].strip(),
                    "answer": card["answer"].strip()
                })
            
            # Return the validated flashcards (limit to requested number)
            return validated_flashcards[:num_flashcards]
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, wait and retry (exponential backoff)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Wait 1s, 2s, 4s...
                time.sleep(wait_time)
                continue
            else:
                # All retries failed - provide helpful error message
                raise ValueError(
                    f"Failed to parse JSON response after {max_retries} attempts. "
                    f"The AI may have returned an invalid format. Error: {str(e)}"
                )
        
        except Exception as e:
            # Handle any other unexpected errors
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                # Provide user-friendly error message
                error_msg = str(e)
                if "rate limit" in error_msg.lower():
                    raise ValueError(
                        "OpenAI API rate limit exceeded. Please wait a moment and try again."
                    )
                elif "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                    raise ValueError(
                        "Invalid API key. Please check your OPENAI_API_KEY environment variable."
                    )
                else:
                    raise ValueError(f"Error generating flashcards: {error_msg}")
    
    # Should not reach here, but included as a safety net
    raise ValueError(f"Failed to generate flashcards after {max_retries} attempts. Please try again.")


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

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application function that handles the UI and user interactions.
    
    This function creates the Streamlit interface with three main sections:
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
                        # Handle API key missing or JSON parsing errors
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
                            st.error(f"⚠️ Error generating flashcards: {error_message}")
                            st.info(
                                "💡 Tip: Try generating again. The AI may have returned "
                                "an invalid response format. If the problem persists, "
                                "try with shorter study material."
                            )
                    
                    except Exception as e:
                        # Handle any other unexpected errors
                        st.error(f"⚠️ An unexpected error occurred: {str(e)}")
                        st.info(
                            "💡 Please try again or check your API key and internet connection. "
                            "If the problem persists, try refreshing the page."
                        )
            
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
                                        
                                        # Initialize quiz state
                                        st.session_state.quiz_mode = True
                                        st.session_state.current_quiz_index = 0
                                        st.session_state.quiz_answers = {}
                                        st.session_state.quiz_scores = {}
                                        st.session_state.quiz_completed = False
                                        st.rerun()
                            
                            except Exception as e:
                                st.error(f"Error starting quiz: {str(e)}")
                                st.info("💡 Please try again or refresh the page.")
                    
                    except Exception as e:
                        st.error(f"Error in quiz setup: {str(e)}")
                
                # Quiz in progress or completed
                else:
                    # Handle edge case: quiz_flashcards might be empty or invalid
                    if (not st.session_state.quiz_flashcards or
                            len(st.session_state.quiz_flashcards) == 0):
                        st.warning("⚠️ Quiz data not found. Please start a new quiz.")
                        if st.button("🔄 Reset Quiz"):
                            try:
                                st.session_state.quiz_mode = False
                                st.session_state.quiz_flashcards = []
                                st.session_state.current_quiz_index = 0
                                st.session_state.quiz_answers = {}
                                st.session_state.quiz_scores = {}
                                st.session_state.quiz_completed = False
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error resetting quiz: {str(e)}")
                    
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
                                # Progress bar
                                try:
                                    progress = (current_index + 1) / total_flashcards
                                    st.progress(progress)
                                    st.caption(f"Question {current_index + 1} of {total_flashcards}")
                                except Exception:
                                    pass  # Silently handle progress bar errors
                                
                                # Get current flashcard safely
                                try:
                                    current_card = st.session_state.quiz_flashcards[current_index]
                                    question = current_card.get('question', 'No question available')
                                    answer = current_card.get('answer', 'No answer available')
                                except (IndexError, KeyError, TypeError) as e:
                                    st.error(f"Error loading question {current_index + 1}: {str(e)}")
                                    st.info("💡 Please reset the quiz and try again.")
                                    if st.button("🔄 Reset Quiz"):
                                        st.session_state.quiz_mode = False
                                        st.session_state.quiz_completed = False
                                        st.rerun()
                                    return
                                
                                # Display current question
                                st.markdown("### Question:")
                                st.info(question)
                                
                                # Answer input area
                                user_answer_key = f"user_answer_{current_index}"
                                if user_answer_key not in st.session_state:
                                    st.session_state[user_answer_key] = ""
                                
                                user_answer = st.text_area(
                                    "Your Answer:",
                                    key=f"answer_input_{current_index}",
                                    height=150,
                                    placeholder="Type your answer here..."
                                )
                                
                                # Navigation buttons
                                col1, col2, col3 = st.columns([1, 1, 1])
                                
                                # Previous button
                                with col1:
                                    if st.button("⏮️ Previous", disabled=(current_index == 0)):
                                        if current_index > 0:
                                            st.session_state.current_quiz_index -= 1
                                            st.rerun()
                                
                                # Submit Answer button
                                with col2:
                                    if st.button("✅ Submit Answer"):
                                        try:
                                            # Save answer
                                            st.session_state.quiz_answers[current_index] = user_answer
                                            st.session_state[user_answer_key] = user_answer
                                            
                                            # Score the answer
                                            similarity, is_correct, keywords = calculate_answer_similarity(
                                                user_answer, answer
                                            )
                                            st.session_state.quiz_scores[current_index] = {
                                                'similarity': similarity,
                                                'is_correct': is_correct,
                                                'keywords': keywords
                                            }
                                            
                                            # Show feedback
                                            if is_correct:
                                                st.success(f"✅ Answer saved! Similarity: {similarity:.1%}")
                                            else:
                                                st.warning(f"⚠️ Answer saved. Similarity: {similarity:.1%}")
                                        except Exception as e:
                                            st.error(f"Error scoring answer: {str(e)}")
                                
                                # Next button
                                with col3:
                                    if st.button("⏭️ Next"):
                                        try:
                                            # Save current answer if not already saved
                                            if current_index not in st.session_state.quiz_answers:
                                                st.session_state.quiz_answers[current_index] = user_answer
                                                st.session_state[user_answer_key] = user_answer
                                                
                                                # Score the answer
                                                similarity, is_correct, keywords = calculate_answer_similarity(
                                                    user_answer, answer
                                                )
                                                st.session_state.quiz_scores[current_index] = {
                                                    'similarity': similarity,
                                                    'is_correct': is_correct,
                                                    'keywords': keywords
                                                }
                                            
                                            # Move to next question or complete quiz
                                            if current_index < total_flashcards - 1:
                                                st.session_state.current_quiz_index += 1
                                                st.rerun()
                                            else:
                                                # Quiz completed
                                                st.session_state.quiz_completed = True
                                                st.rerun()
                                        except Exception as e:
                                            st.error(f"Error moving to next question: {str(e)}")
                                
                                # Optional: Show correct answer checkbox
                                if st.checkbox("Show Correct Answer", key=f"show_correct_{current_index}"):
                                    st.markdown("**Correct Answer:**")
                                    st.success(answer)
                                    
                                    # Show scoring info if answer was submitted
                                    if current_index in st.session_state.quiz_scores:
                                        try:
                                            score_info = st.session_state.quiz_scores[current_index]
                                            similarity = score_info.get('similarity', 0)
                                            is_correct = score_info.get('is_correct', False)
                                            status = '✅ Correct' if is_correct else '❌ Needs Improvement'
                                            st.caption(f"Similarity: {similarity:.1%} | Status: {status}")
                                        except Exception:
                                            pass
                            
                            # Quiz completed - show results
                            elif st.session_state.quiz_completed:
                                st.balloons()  # Celebration animation
                                st.success("🎉 Quiz Completed!")
                                
                                try:
                                    # Calculate final score
                                    total = len(st.session_state.quiz_flashcards)
                                    correct_count = 0
                                    incorrect_answers = []
                                    
                                    # Score all answers that weren't scored during quiz
                                    for idx in range(total):
                                        try:
                                            # Get user answer safely
                                            user_ans = st.session_state.quiz_answers.get(idx, "")
                                            
                                            # Get flashcard safely
                                            if idx >= len(st.session_state.quiz_flashcards):
                                                continue
                                            card = st.session_state.quiz_flashcards[idx]
                                            correct_ans = card.get('answer', '')
                                            
                                            # Score if not already scored
                                            if idx not in st.session_state.quiz_scores:
                                                similarity, is_correct, keywords = calculate_answer_similarity(
                                                    user_ans, correct_ans
                                                )
                                                st.session_state.quiz_scores[idx] = {
                                                    'similarity': similarity,
                                                    'is_correct': is_correct,
                                                    'keywords': keywords
                                                }
                                            
                                            # Count correct/incorrect
                                            score_info = st.session_state.quiz_scores.get(idx, {})
                                            if score_info.get('is_correct', False):
                                                correct_count += 1
                                            else:
                                                incorrect_answers.append({
                                                    'index': idx,
                                                    'card': card,
                                                    'user_answer': user_ans,
                                                    'similarity': score_info.get('similarity', 0)
                                                })
                                        except Exception as e:
                                            # If one answer fails, continue with others
                                            st.warning(f"Error processing answer {idx + 1}: {str(e)}")
                                            continue
                                    
                                    # Display final score metrics
                                    try:
                                        percentage = (correct_count / total * 100) if total > 0 else 0
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.metric("Correct Answers", f"{correct_count}/{total}")
                                        with col2:
                                            st.metric("Score", f"{percentage:.1f}%")
                                        with col3:
                                            st.metric("Incorrect", len(incorrect_answers))
                                    except Exception:
                                        pass
                                    
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
                                                card = item.get('card', {})
                                                user_ans = item.get('user_answer', '')
                                                similarity = item.get('similarity', 0)
                                                
                                                question = card.get('question', 'No question')[:60]
                                                with st.expander(
                                                    f"❌ Question {idx + 1}: {question}... (Similarity: {similarity:.1%})",
                                                    expanded=True
                                                ):
                                                    st.markdown("**❓ Question:**")
                                                    st.info(card.get('question', 'N/A'))
                                                    
                                                    st.markdown("**✍️ Your Answer:**")
                                                    st.warning(user_ans if user_ans else "No answer provided")
                                                    
                                                    st.markdown("**✅ Correct Answer:**")
                                                    st.success(card.get('answer', 'N/A'))
                                                    
                                                    if similarity > 0:
                                                        st.caption(
                                                            f"Similarity Score: {similarity:.1%} - "
                                                            "Your answer was close but needs more detail."
                                                        )
                                            except Exception:
                                                continue
                                    else:
                                        st.success("🎊 Perfect! You got all answers correct!")
                                    
                                    # Show all answers summary
                                    st.subheader("📋 Complete Answer Review")
                                    for idx, card in enumerate(st.session_state.quiz_flashcards):
                                        try:
                                            user_ans = st.session_state.quiz_answers.get(idx, "")
                                            score_info = st.session_state.quiz_scores.get(idx, {})
                                            is_correct = score_info.get('is_correct', False)
                                            
                                            status_icon = "✅" if is_correct else "❌"
                                            similarity = score_info.get('similarity', 0)
                                            similarity_text = f" (Similarity: {similarity:.1%})" if score_info else ""
                                            
                                            question = card.get('question', 'No question')[:50]
                                            with st.expander(
                                                f"{status_icon} Question {idx + 1}: {question}...{similarity_text}"
                                            ):
                                                st.markdown("**Question:**")
                                                st.write(card.get('question', 'N/A'))
                                                
                                                st.markdown("**Your Answer:**")
                                                st.write(user_ans if user_ans else "No answer provided")
                                                
                                                st.markdown("**Correct Answer:**")
                                                st.write(card.get('answer', 'N/A'))
                                                
                                                if score_info:
                                                    st.caption(f"Similarity: {score_info.get('similarity', 0):.1%}")
                                        except Exception:
                                            continue
                                    
                                    # Reset quiz button
                                    col1, col2, col3 = st.columns([1, 1, 1])
                                    with col2:
                                        if st.button("🔄 Take Quiz Again", type="primary", use_container_width=True):
                                            try:
                                                st.session_state.quiz_mode = False
                                                st.session_state.quiz_flashcards = []
                                                st.session_state.current_quiz_index = 0
                                                st.session_state.quiz_answers = {}
                                                st.session_state.quiz_scores = {}
                                                st.session_state.quiz_completed = False
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
