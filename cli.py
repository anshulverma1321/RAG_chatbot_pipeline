#!/usr/bin/env python3
import os
# Set Hugging Face mirror endpoint at the absolute start to prevent connection issues in sub-libraries
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import argparse
from dotenv import load_dotenv

# Ensure root folder is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

from RAG.db import init_db, list_documents
from RAG.vector_store import init_vector_store
from RAG.ingestion import process_pdf
from RAG.query_engine import execute_rag_query

# Compute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.join(BASE_DIR, "RAG")
DATA_DIR = os.path.join(RAG_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "rag_tool.db")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "qdrant")
TEMP_DIR = os.path.join(DATA_DIR, "temp")

# ANSI Colors
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_CYAN = "\033[36m"
C_WHITE = "\033[37m"

def print_header(title: str):
    print(f"\n{C_BOLD}{C_CYAN}=== {title} ==={C_RESET}")

def print_error(msg: str):
    print(f"{C_BOLD}{C_RED}Error:{C_RESET} {msg}")

def print_success(msg: str):
    print(f"{C_BOLD}{C_GREEN}Success:{C_RESET} {msg}")

def check_api_key():
    if not os.environ.get("GEMINI_API_KEY"):
        print(f"{C_BOLD}{C_YELLOW}Warning: GEMINI_API_KEY is not set in your environment or .env file.{C_RESET}")
        print("Queries and PDF ingestion (VLM image summarization) will fail without it.\n")

def resolve_pdf_paths(input_str: str) -> tuple[list[str], list[str]]:
    """
    Resolves a comma-separated list of filenames or paths.
    Searches in:
      1. Absolute path check
      2. Current Working Directory (CWD)
      3. Project Root (BASE_DIR)
      4. Data Directory (DATA_DIR)
    Also tries appending '.pdf' if not present.
    Returns (resolved_paths, unresolved_names).
    """
    parts = [p.strip().strip('\'"') for p in input_str.split(',') if p.strip()]
    resolved = []
    unresolved = []
    
    search_dirs = [
        os.getcwd(),
        BASE_DIR,
        DATA_DIR
    ]
    
    for part in parts:
        if not part:
            continue
            
        found_path = None
        
        # 1. Absolute path check
        if os.path.isabs(part):
            if os.path.exists(part) and os.path.isfile(part):
                found_path = part
        else:
            # 2. Check each search directory
            for s_dir in search_dirs:
                candidate = os.path.join(s_dir, part)
                if os.path.exists(candidate) and os.path.isfile(candidate):
                    found_path = candidate
                    break
            
            # 3. Try appending .pdf if not present
            if not found_path and not part.lower().endswith(".pdf"):
                part_pdf = part + ".pdf"
                for s_dir in search_dirs:
                    candidate = os.path.join(s_dir, part_pdf)
                    if os.path.exists(candidate) and os.path.isfile(candidate):
                        found_path = candidate
                        break
                        
        if found_path:
            resolved.append(found_path)
        else:
            unresolved.append(part)
            
    return resolved, unresolved

def get_doc_ids_from_names(filenames: list[str]) -> list[int]:
    """Resolves a list of filenames/filters to internal database IDs."""
    if not filenames:
        return []
    try:
        docs = list_documents(DB_PATH)
    except Exception as e:
        print_error(f"Failed to check database: {e}")
        return []
        
    doc_map = {d['filename'].lower(): d['id'] for d in docs}
    ids = []
    for name in filenames:
        name_lower = name.lower().strip()
        if name_lower in doc_map:
            ids.append(doc_map[name_lower])
        else:
            # Attempt partial matching
            matched = False
            for db_name, db_id in doc_map.items():
                if name_lower in db_name or name_lower in os.path.basename(db_name).lower():
                    ids.append(db_id)
                    matched = True
                    break
            if not matched:
                print(f"{C_BOLD}{C_YELLOW}Warning: Document '{name}' not found. Ignoring filter.{C_RESET}")
    return ids

def handle_query(query_text: str, doc_ids: list = None):
    print(f"\n{C_BLUE}Querying: \"{query_text}\"...{C_RESET}")
    try:
        answer = execute_rag_query(
            query=query_text,
            db_path=DB_PATH,
            vector_db_path=VECTOR_DB_PATH,
            document_ids=doc_ids,
            top_k=5
        )
        print(f"\n{C_BOLD}{C_GREEN}Answer:{C_RESET}")
        print(answer)
        print()
    except Exception as e:
        from RAG.logger import log_error
        log_error("cli.py", "handle_query", type(e).__name__, str(e))
        print_error(f"Query execution failed: {e}")

def process_and_print_query(query_text: str) -> str:
    print(f"\n{C_BOLD}{C_GREEN}Assistant:{C_RESET}")
    try:
        answer = execute_rag_query(
            query=query_text,
            db_path=DB_PATH,
            vector_db_path=VECTOR_DB_PATH,
            document_ids=None,
            top_k=5
        )
        print(answer)
        print()
        
        # Trigger Text-to-Speech playback
        from RAG.text_to_speech import TextToSpeechEngine
        tts = TextToSpeechEngine.get_instance()
        if tts.available and tts.voice_enabled:
            print(f"🔊 {C_BLUE}Speaking response...{C_RESET}")
            tts.speak(answer)
            
        return answer
    except Exception as e:
        from RAG.logger import log_error
        log_error("cli.py", "process_and_print_query", type(e).__name__, str(e))
        print_error(f"Failed to generate answer: {e}")
        return ""

def run_voice_input_flow(temp_wav_path: str, transcriber) -> str:
    """Helper to run the speech recording, transcription, and confirmation loop."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    from RAG.speech_to_text import record_audio, SpeechTranscriber
    
    while True:
        success = record_audio(temp_wav_path)
        if not success:
            print_error("Audio recording failed or was interrupted. Check your microphone connection.")
            return ""
            
        print(f"\n{C_BLUE}Transcribing speech using Whisper...{C_RESET}")
        try:
            if transcriber is None:
                transcriber = SpeechTranscriber.get_instance(model_size="tiny")
            recognized_text = transcriber.transcribe(temp_wav_path)
        except Exception as e:
            print_error(f"Failed to transcribe speech: {e}")
            return ""
        finally:
            if os.path.exists(temp_wav_path):
                try:
                    os.remove(temp_wav_path)
                except Exception:
                    pass
                    
        if not recognized_text:
            print_error("Could not understand speech (empty transcription).")
            retry = input("Try speaking again? (y/n): ").strip().lower()
            if retry == 'y':
                continue
            else:
                return ""
                
        print(f"\n{C_BOLD}{C_CYAN}Recognized Question:{C_RESET}")
        print(f"\"{recognized_text}\"")
        
        # Ask for confirmation
        conf_choice = None
        while True:
            print(f"\n{C_BOLD}Use this question?{C_RESET}")
            print("1. Yes")
            print("2. Record Again")
            print("3. Cancel")
            try:
                conf_choice = input(f"{C_BOLD}Enter choice (1-3): {C_RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                conf_choice = '3'
                break
                
            if conf_choice in ['1', '2', '3']:
                break
            else:
                print_error("Invalid option. Enter 1, 2, or 3.")
                
        if conf_choice == '1':
            return recognized_text
        elif conf_choice == '2':
            continue # loop back to record audio
        else:
            return "" # Cancel voice flow

def enter_chat_mode():
    try:
        docs = list_documents(DB_PATH)
    except Exception as e:
        print_error(f"Failed to load document list: {e}")
        return
        
    if not docs:
        print(f"\n{C_YELLOW}No documents are currently loaded in the database.{C_RESET}")
        print(f"Please upload PDF documents first (Option 1).{C_RESET}")
        return
        
    # 1. Voice Settings prompt
    print(f"\n{C_BOLD}{C_CYAN}===================================================={C_RESET}")
    print(f"{C_BOLD}{C_CYAN}                   Voice Settings                   {C_RESET}")
    print(f"{C_BOLD}{C_CYAN}===================================================={C_RESET}")
    print("Enable Voice Responses?")
    print("1. Yes")
    print("2. No")
    
    voice_enabled = True
    try:
        voice_choice = input(f"{C_BOLD}Enter choice (1-2): {C_RESET}").strip()
        if voice_choice == '2':
            voice_enabled = False
    except (KeyboardInterrupt, EOFError):
        print("\nUsing defaults (Voice Enabled).")
        
    # Initialize TTS Engine and update setting
    from RAG.text_to_speech import TextToSpeechEngine
    tts_engine = TextToSpeechEngine.get_instance()
    tts_engine.voice_enabled = voice_enabled
    
    if voice_enabled and not tts_engine.available:
        print(f"{C_RED}Voice playback unavailable (missing Piper or voice model).{C_RESET}")
        print("Displaying text response only.")
        
    # 2. Display Loaded Documents
    print(f"\n{C_BOLD}{C_CYAN}Documents Loaded:{C_RESET}")
    for d in docs:
        print(f"- {d['filename']}")
        
    print(f"\n{C_BOLD}{C_CYAN}===================================================={C_RESET}")
    print(f"{C_BOLD}{C_CYAN}                    CHAT MODE                       {C_RESET}")
    print(f"{C_BOLD}{C_CYAN}===================================================={C_RESET}")
    
    # Path for temporary spoken input
    temp_wav_path = os.path.join(TEMP_DIR, "voice_input.wav")
    
    # Lazy load transcriber
    transcriber = None
    
    # Initial menu selection state
    initial_loop = True
    while True:
        if initial_loop:
            print(f"\n{C_BOLD}Choose Input Method:{C_RESET}")
            print("1. Type Question")
            print("2. Speak Question")
            print("3. Exit Chat Mode")
            
            try:
                choice = input(f"{C_BOLD}Enter choice (1-3): {C_RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                break
                
            if choice == '1':
                try:
                    query_text = input(f"\n{C_BOLD}Type your question:{C_RESET}\n").strip()
                except (KeyboardInterrupt, EOFError):
                    continue
                if not query_text:
                    continue
                if query_text.lower() in ['exit', 'quit', 'back']:
                    break
                tts_engine.stop()
                process_and_print_query(query_text)
                initial_loop = False
                
            elif choice == '2':
                tts_engine.stop()
                recognized = run_voice_input_flow(temp_wav_path, transcriber)
                if recognized:
                    process_and_print_query(recognized)
                    initial_loop = False
                    
            elif choice == '3':
                break
            else:
                print_error("Invalid option. Enter 1, 2, or 3.")
        else:
            # Continuous Options Loop after receiving first response
            print(f"\n{C_BOLD}{C_CYAN}===================================================={C_RESET}")
            print(f"{C_BOLD}{C_CYAN}                     Options                        {C_RESET}")
            print(f"{C_BOLD}{C_CYAN}===================================================={C_RESET}")
            print("1. Type Another Question")
            print("2. Speak Another Question")
            print("3. Repeat Last Answer")
            status_str = "Enabled" if tts_engine.voice_enabled else "Disabled"
            print(f"4. Toggle Voice Responses (Currently: {status_str})")
            print("5. Exit Chat Mode")
            print(f"{C_BOLD}{C_CYAN}===================================================={C_RESET}")
            
            try:
                opt_choice = input(f"{C_BOLD}Enter choice (1-5): {C_RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                break
                
            if opt_choice == '1':
                try:
                    query_text = input(f"\n{C_BOLD}Type your question:{C_RESET}\n").strip()
                except (KeyboardInterrupt, EOFError):
                    continue
                if not query_text:
                    continue
                if query_text.lower() in ['exit', 'quit', 'back']:
                    break
                tts_engine.stop()
                process_and_print_query(query_text)
                
            elif opt_choice == '2':
                tts_engine.stop()
                recognized = run_voice_input_flow(temp_wav_path, transcriber)
                if recognized:
                    process_and_print_query(recognized)
                    
            elif opt_choice == '3':
                print(f"🔊 {C_BLUE}Repeating last answer...{C_RESET}")
                tts_engine.repeat()
                
            elif opt_choice == '4':
                tts_engine.voice_enabled = not tts_engine.voice_enabled
                if not tts_engine.voice_enabled:
                    tts_engine.stop()
                status_str = "Enabled" if tts_engine.voice_enabled else "Disabled"
                print(f"{C_GREEN}[Voice Responses {status_str}]{C_RESET}")
                
            elif opt_choice == '5':
                tts_engine.stop()
                break
            else:
                print_error("Invalid option. Enter a choice between 1 and 5.")
                
    tts_engine.stop()

def interactive_loop():
    check_api_key()
    
    while True:
        print(f"\n{C_BOLD}{C_CYAN}===================================================={C_RESET}")
        print(f"{C_BOLD}{C_CYAN}     Multimodal Document Intelligence RAG CLI       {C_RESET}")
        print(f"{C_BOLD}{C_CYAN}===================================================={C_RESET}")
        print(f" 1. Ingest / Upload PDF Documents")
        print(f" 2. Chat with Documents")
        print(f" 3. Exit")
        print(f"{C_BOLD}{C_CYAN}===================================================={C_RESET}")
        
        try:
            choice = input(f"{C_BOLD}Enter choice (1-3): {C_RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if choice == '1':
            print_header("Ingest / Upload PDF Documents")
            print("Enter PDF file names or paths (comma-separated):")
            input_str = input("> ").strip()
            if not input_str:
                continue
                
            resolved_paths, unresolved_names = resolve_pdf_paths(input_str)
            
            if unresolved_names:
                for name in unresolved_names:
                    print_error(f"Could not find file: {name}")
            
            if not resolved_paths:
                print(f"{C_YELLOW}No valid PDF files found to ingest.{C_RESET}")
                continue
                
            success_count = 0
            for path in resolved_paths:
                filename = os.path.basename(path)
                print(f"{C_BLUE}Ingesting {filename}...{C_RESET}")
                try:
                    doc_id, msg = process_pdf(path, DB_PATH, VECTOR_DB_PATH)
                    print(f"{C_BOLD}{C_GREEN}[OK] {filename} uploaded: {msg}{C_RESET}")
                    success_count += 1
                except Exception as e:
                    print_error(f"Failed to ingest {filename}: {e}")
                    
            if success_count > 0:
                print(f"\n{C_BOLD}{C_GREEN}Ingestion complete. Entering Chat Mode...{C_RESET}")
                enter_chat_mode()
                
        elif choice == '2':
            enter_chat_mode()
            
        elif choice == '3':
            print("Exiting. Goodbye!")
            break
        else:
            print_error("Invalid option. Please choose between 1 and 3.")

def main():
    try:
        # Initialize Databases on startup
        init_db(DB_PATH)
        init_vector_store(VECTOR_DB_PATH)
        
        # Run startup system diagnostics
        from RAG.logger import log_system_check
        log_system_check(DB_PATH, VECTOR_DB_PATH)
    except Exception as e:
        print_error(f"Startup initialization failed: {e}")
        from RAG.logger import log_error
        log_error("cli.py", "main_startup", type(e).__name__, str(e))
        sys.exit(1)
    
    # Setup argparse
    parser = argparse.ArgumentParser(
        description="Multimodal Document Intelligence RAG CLI - Chat, upload, list, or delete PDF documents."
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Ingest subcommand
    parser_ingest = subparsers.add_parser("upload", help="Ingest local PDF documents")
    parser_ingest.add_argument("file_paths", type=str, help="Comma-separated absolute/relative paths or filenames to PDF files")

    # Query subcommand
    parser_query = subparsers.add_parser("query", help="Run a grounded search query")
    parser_query.add_argument("query_text", type=str, help="Question/Search query")
    parser_query.add_argument("--doc-names", nargs="+", type=str, help="Optional filenames to restrict search to")

    # Interactive subcommand (also default)
    subparsers.add_parser("interactive", help="Start the conversational CLI loop")

    args = parser.parse_args()

    if args.command == "upload":
        check_api_key()
        resolved_paths, unresolved_names = resolve_pdf_paths(args.file_paths)
        if unresolved_names:
            for name in unresolved_names:
                print_error(f"Could not find file: {name}")
        if resolved_paths:
            for path in resolved_paths:
                filename = os.path.basename(path)
                print(f"{C_BLUE}Ingesting {filename}...{C_RESET}")
                try:
                    doc_id, msg = process_pdf(path, DB_PATH, VECTOR_DB_PATH)
                    print(f"{C_BOLD}{C_GREEN}[OK] {filename} uploaded: {msg}{C_RESET}")
                except Exception as e:
                    print_error(f"Failed to ingest {filename}: {e}")
                    
    elif args.command == "query":
        check_api_key()
        doc_ids = None
        if args.doc_names:
            doc_ids = get_doc_ids_from_names(args.doc_names)
        handle_query(args.query_text, doc_ids)
        
    elif args.command == "interactive" or args.command is None:
        interactive_loop()

if __name__ == "__main__":
    main()
