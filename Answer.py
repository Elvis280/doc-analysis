"""
Answer.py: High-level QA pipeline. Includes Query Understanding (Entity Extraction), 
Context Context Construction (Row Filtering), and LLM Answer Generation (Groq).
"""

import os
import json
from typing import List, Dict, Optional, Any, Union, Tuple
import re
from datetime import datetime

from Retrieval import run_rag_pipeline

from dotenv import load_dotenv

load_dotenv()

from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set in environment variables")

client = Groq(api_key=GROQ_API_KEY)


# -------------------------------------------------------
# CONTEXT BUILDER
# -------------------------------------------------------

# -------------------------------------------------------
# NORMALIZATION & RANGE LOGIC
# -------------------------------------------------------

def parse_date(text: str) -> Optional[datetime]:
    """Attempts to parse a date string into a datetime object."""
    # Common formats: DD.MM.YYYY, DD/MM/YYYY, YYYY-MM-DD, DD-MM-YYYY
    formats = [
        "%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y",
        "%d %b %Y", "%d %B %Y" # 07 Jan 2026
    ]
    norm_text = text.strip()
    # Remove ordinal suffixes (e.g., 7th, 1st) - simple regex
    norm_text = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', norm_text)
    
    for fmt in formats:
        try:
            return datetime.strptime(norm_text, fmt)
        except ValueError:
            continue
    return None

def parse_number(text: str) -> Optional[float]:
    """Attempts to parse a number (int or float) from text."""
    # Removes commas, handles simple floats
    clean = text.replace(",", "").strip()
    try:
        return float(clean)
    except ValueError:
        return None

def parse_range(text: str) -> Optional[Tuple[Any, Any]]:
    """
    Detects if text is a range (numeric or date).
    Returns (start_val, end_val) if detected, else None.
    Supports: "A - B", "A to B", "A – B"
    """
    # Splitters: hyphen, en-dash, "to"
    # Note: Regex allows optional spaces around splitters
    parts = re.split(r'\s*(?:-|–|to)\s*', text.strip())
    
    if len(parts) != 2:
        return None
        
    start_str, end_str = parts[0], parts[1]
    
    # Try Numeric Range
    n1, n2 = parse_number(start_str), parse_number(end_str)
    if n1 is not None and n2 is not None:
        return (n1, n2)
        
    # Try Date Range
    d1, d2 = parse_date(start_str), parse_date(end_str)
    if d1 and d2:
        return (d1, d2)
        
    # Fallback: maybe mixed? assume no if types don't match
    return None

def is_match(entity_key: str, target_val: Any, cell_text: str) -> bool:
    """
    Determines if a cell matches the entity.
    target_val: The parsed value of the entity (float for roll_no, datetime for date, string for subject).
    """
    cell_clean = str(cell_text).strip()
    
    # 1. Direct Text Match (Case-insensitive inclusion) for robust fallback
    # Note: target_val might be float/datetime, so convert to str for text checks
    if str(target_val).lower() in cell_clean.lower():
        return True
    
    # 2. Logic based on Entity Type
    
    # -- ROLL NUMBER (Numeric) --
    if entity_key == "roll_no":
        if not isinstance(target_val, float):
             return False

        # Check if cell is a range
        rng = parse_range(cell_clean)
        if rng and isinstance(rng[0], float):
            # Numeric range
            start, end = rng
            if start <= target_val <= end:
                return True
        
        # Check if cell is a single number 
        cell_num = parse_number(cell_clean)
        if cell_num is not None:
             if cell_num == target_val:
                 return True

    # -- DATE (Datetime) --
    elif entity_key == "date":
        if not isinstance(target_val, datetime):
            return False
             
        # Check if cell is a range
        rng = parse_range(cell_clean)
        if rng and isinstance(rng[0], datetime):
            start, end = rng
            if start <= target_val <= end:
                return True
                
        # Check if cell is single date
        cell_date = parse_date(cell_clean)
        if cell_date:
            if cell_date == target_val:
                return True

    # -- AMOUNT/FEES (Numeric) --
    elif entity_key == "amount":
        if not isinstance(target_val, float):
            return False
             
        rng = parse_range(cell_clean)
        if rng and isinstance(rng[0], float):
             if rng[0] <= target_val <= rng[1]:
                 return True

    return False

def parse_year(text: str) -> Optional[int]:
    """
    Normalizes academic year strings to integers.
    Handles: "1st", "I", "First", "1", "2nd", "II", etc.
    Returns integer year (1-5) or None.
    """
    text = text.lower().strip()
    
    # 1. Roman Numerals
    roman_map = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}
    if text in roman_map:
        return roman_map[text]
        
    # 2. Ordinals/Words
    word_map = {
        "1st": 1, "first": 1, 
        "2nd": 2, "second": 2, 
        "3rd": 3, "third": 3, 
        "4th": 4, "fourth": 4, 
        "5th": 5, "fifth": 5
    }
    # Check exact match first
    if text in word_map:
        return word_map[text]
    
    # Check regex for "1st", "2nd" etc if attached to other chars? 
    # Usually passed clean, but "1st" from "1st year"
    match = re.match(r'^(\d+)(st|nd|rd|th)$', text)
    if match:
        return int(match.group(1))
        
    # 3. Simple digits
    if text.isdigit():
        return int(text)
        
    return None

def extract_years_from_text(text: str) -> List[int]:
    """
    Finds all year mentions in a text string.
    Looks for: I, II, III, IV, 1st, 2nd, 3rd...
    Context aware: usually followed by "year" or comma separated in a "year" column.
    We'll be generous: find any token that looks like a year index.
    """
    # Split by comma or space
    tokens = re.split(r'[,\s/]+', text)
    years = []
    for t in tokens:
        y = parse_year(t)
        if y is not None and 1 <= y <= 5: # Academic years usually 1-5
            years.append(y)
    return years


def get_row_numbers(row: List[Any]) -> List[float]:
    """Extracts all parseable numbers from a row using regex."""
    nums = []
    for cell in row:
        matches = re.findall(r'\d+(?:\.\d+)?', str(cell))
        for m in matches:
             try:
                 nums.append(float(m))
             except ValueError:
                 pass
    return nums

def get_row_dates(row: List[Any]) -> List[datetime]:
    """Extracts all parseable dates from a row."""
    dates = []
    for cell in row:
        # Try parse whole cell first
        val = parse_date(str(cell))
        if val:
            dates.append(val)
        else:
             parts = str(cell).split()
             for p in parts:
                 v = parse_date(p)
                 if v:
                     dates.append(v)
    return dates

def filter_table_rows(rows: List[List[Any]], entities: Dict[str, Any]) -> List[List[Any]]:
    """
    Filters rows based on extracted entities.
    Expects entities dict to have parsed values (e.g., "roll_no_val": float).
    """
    if not entities:
        return rows

    filtered_rows = []
    for i, row in enumerate(rows):
        # Always keep header
        if i == 0:
            filtered_rows.append(row)
            continue
        
        row_matched = False
        
        # 1. Roll Number Filter
        if "roll_no_val" in entities:
            target_roll = entities["roll_no_val"]
            
            # A. Check Cell-Level Match
            for cell in row:
                if is_match("roll_no", target_roll, str(cell)):
                    row_matched = True
                    break
            
            # B. Check Row-Level Implicit Range
            if not row_matched:
                row_nums = get_row_numbers(row)
                valid_nums = [n for n in row_nums if n > 1000000] # Heuristic for roll nos
                if len(valid_nums) >= 2:
                    if min(valid_nums) <= target_roll <= max(valid_nums):
                        row_matched = True

        # 2. Date Filter
        elif "date_val" in entities:
             target_date = entities["date_val"]
             
             # A. Check Cell-Level Match
             for cell in row:
                  if is_match("date", target_date, str(cell)):
                      row_matched = True
                      break
            
             # B. Check Row-Level Implicit Range
             if not row_matched:
                 row_dates = get_row_dates(row)
                 if len(row_dates) >= 2:
                     if min(row_dates) <= target_date <= max(row_dates):
                         row_matched = True
        
        # 3. Academic Year Filter (New)
        # Applied before subject because it's a structural filter.
        # If 'year_val' is in entities, we strictly filter matching rows.
        # Matches if the ROW contains the year (e.g. "I year") or a list containing it ("I, II year")
        elif "year_val" in entities:
            target_year = entities["year_val"]
            # Scan whole row for year mentions
            found_years = set()
            for cell in row:
                cell_years = extract_years_from_text(str(cell))
                found_years.update(cell_years)
            
            if target_year in found_years:
                row_matched = True

        # 4. Subject Filter
        elif "subject" in entities:
             target_subject = entities["subject"]
             row_text = " ".join(str(cell) for cell in row).lower()
             if target_subject.lower() in row_text:
                 row_matched = True
        
        if row_matched:
            filtered_rows.append(row)
            
    return filtered_rows


def build_context(results: List[Dict], entities: Optional[Dict[str, Any]] = None) -> str:
    """
    Converts retrieved chunks into LLM-readable context with row filtering.
    """
    if entities is None:
        entities = {}

    context_blocks = []

    for r in results:
        header = (
            f"[Chunk ID: {r['chunk_id']} | "
            f"Page: {r.get('page')} | "
            f"Type: {r['chunk_type']} | "
            f"Intent: {r.get('intent', 'N/A')}]"
        )

        if r["chunk_type"] == "table_generic":
            rows = r["raw_chunk"]["table"].get("rows", [])
            
            # Use helper to filter
            final_rows = filter_table_rows(rows, entities)

            if not final_rows:
                 continue

            table_text = "\n".join(
                " | ".join(str(cell) for cell in row) for row in final_rows
            )

            block = (
                f"{header}\n"
                f"TABLE DATA:\n{table_text}"
            )

        else:
            block = (
                f"{header}\n"
                f"TEXT:\n{r['text']}"
            )

        context_blocks.append(block)

    return "\n\n---\n\n".join(context_blocks)


# -------------------------------------------------------
# CHUNK OPTIMIZATION FUNCTIONS
# -------------------------------------------------------

def hash_table_rows(rows: List[List[Any]]) -> str:
    """Create a hash of table rows for de-duplication."""
    # Convert to string and hash
    content = "|".join("|".join(str(cell) for cell in row) for row in rows)
    return str(hash(content))


def estimate_chunk_tokens(chunk: Dict) -> int:
    """
    Estimate token count for a chunk.
    Rough estimate: ~4 characters per token.
    """
    if chunk['chunk_type'] == 'table_generic':
        rows = chunk['raw_chunk']['table'].get('rows', [])
        # Table format: "cell | cell | cell\n"
        text = "\n".join(" | ".join(str(cell) for cell in row) for row in rows)
    else:
        text = chunk.get('text', '')
    
    # Add metadata overhead
    metadata_chars = 100  # [Chunk ID: ... | Page: ... | Type: ...]
    total_chars = len(text) + metadata_chars
    
    return total_chars // 4  # ~4 chars per token


def calculate_entity_match_score(chunk: Dict, entities: Dict[str, Any]) -> float:
    """
    Calculate how well a chunk matches extracted entities.
    Returns score between 0.0 and 1.0.
    """
    if not entities:
        return 0.5  # Neutral score if no entities
    
    # For text chunks, do simple keyword matching
    if chunk['chunk_type'] != 'table_generic':
        text = chunk.get('text', '').lower()
        matches = 0
        total_entities = 0
        
        for key, value in entities.items():
            if key.startswith('_'):  # Skip internal keys
                continue
            total_entities += 1
            if str(value).lower() in text:
                matches += 1
        
        return matches / max(total_entities, 1)
    
    # For table chunks, check filtered row count
    rows = chunk['raw_chunk']['table'].get('rows', [])
    if not rows or len(rows) <= 1:  # Empty or header-only
        return 0.0
    
    # Filter rows with entities
    filtered_rows = filter_table_rows(rows, entities)
    
    # Score based on match ratio (exclude header)
    total_rows = len(rows) - 1  # Exclude header
    matched_rows = len(filtered_rows) - 1 if filtered_rows else 0
    
    if total_rows == 0:
        return 0.0
    
    return matched_rows / total_rows


def rerank_chunks(chunks: List[Dict], entities: Dict[str, Any]) -> List[Dict]:
    """
    Re-rank chunks based on combined vector similarity and entity match scores.
    """
    for chunk in chunks:
        # Vector similarity score (distance is 0-2, convert to 0-1 similarity)
        vector_score = max(0, 1 - (chunk.get('distance', 1.0) / 2.0))
        
        # Entity match score (0-1)
        entity_score = chunk.get('entity_score', 0.0)
        
        # Weighted combination (favor entity matches slightly)
        chunk['combined_score'] = (0.4 * vector_score) + (0.6 * entity_score)
    
    # Sort by combined score (highest first)
    return sorted(chunks, key=lambda x: x.get('combined_score', 0), reverse=True)


def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Remove chunks with duplicate content.
    """
    seen_hashes = set()
    unique_chunks = []
    
    for chunk in chunks:
        # Create content hash
        if chunk['chunk_type'] == 'table_generic':
            rows = chunk['raw_chunk']['table'].get('rows', [])
            content_hash = hash_table_rows(rows)
        else:
            content_hash = str(hash(chunk.get('text', '')))
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_chunks.append(chunk)
    
    return unique_chunks


def ensure_diversity(chunks: List[Dict], max_per_page: int = 3) -> List[Dict]:
    """
    Ensure chunks come from diverse pages.
    Limits number of chunks per page to prevent page-specific bias.
    """
    page_counts = {}
    diverse_chunks = []
    
    for chunk in chunks:
        page = chunk.get('page', 0)
        count = page_counts.get(page, 0)
        
        if count < max_per_page:
            diverse_chunks.append(chunk)
            page_counts[page] = count + 1
    
    return diverse_chunks


def optimize_for_token_budget(
    chunks: List[Dict],
    max_tokens: int = 2500
) -> List[Dict]:
    """
    Select best chunks within token budget.
    Assumes chunks are already sorted by relevance.
    """
    selected = []
    current_tokens = 0
    
    for chunk in chunks:
        chunk_tokens = estimate_chunk_tokens(chunk)
        
        if current_tokens + chunk_tokens > max_tokens:
            # Stop if budget exceeded
            break
        
        selected.append(chunk)
        current_tokens += chunk_tokens
    
    return selected


def optimize_chunks(
    chunks: List[Dict],
    entities: Dict[str, Any],
    max_tokens: int = 2500,
    max_per_page: int = 3
) -> List[Dict]:
    """
    Complete chunk optimization pipeline.
    
    Steps:
    1. Calculate entity match scores
    2. Re-rank by combined score
    3. De-duplicate content
    4. Ensure page diversity
    5. Apply token budget
    
    Returns optimized list of chunks.
    """
    if not chunks:
        return []
    
    # 1. Calculate entity scores for each chunk
    for chunk in chunks:
        chunk['entity_score'] = calculate_entity_match_score(chunk, entities)
    
    # 2. Re-rank based on combined scores
    chunks = rerank_chunks(chunks, entities)
    
    # 3. Remove duplicates
    chunks = deduplicate_chunks(chunks)
    
    # 4. Ensure diversity across pages
    chunks = ensure_diversity(chunks, max_per_page=max_per_page)
    
    # 5. Apply token budget
    chunks = optimize_for_token_budget(chunks, max_tokens=max_tokens)
    
    return chunks



# -------------------------------------------------------
# ANSWER GENERATION
# -------------------------------------------------------

def generate_answer(
    query: str,
    context: str,
    model: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Uses Groq LLM to generate final answer.
    """
    system_prompt = (
        '''You are an academic document assistant specialized in handling OCR-processed documents.

            You MUST answer strictly using the provided context.
            Do NOT use outside knowledge.

            Core constraints:
            - If the answer is not present in the context, clearly state that it is not available.
            - Do NOT invent information that doesn't exist in any form in the context.
            - Reason within the document's established patterns and structure.

            ENHANCED OCR CORRECTION RULES:
            Since the data comes from OCR, you MAY intelligently correct OCR errors and fill gaps 
            when there is CLEAR CONTEXTUAL EVIDENCE within the same document.

            Allowed OCR corrections and gap filling:
            
            1. **Pattern-based inference (RECOMMENDED)**:
               - If a table shows "DBMS - 15th Jan", "OS - 16th Jan", "CN - ?th Jan"
                 AND the pattern suggests sequential dates, you MAY infer the missing date
                 IF it fits the established pattern in the SAME table.
               
            2. **OCR artifact correction (ALWAYS SAFE)**:
               - "27H January 2026" → "27th January 2026"
               - "1l" or "Il" or "I0" → "II" (Roman numerals)
               - "O" (letter) → "0" (number) in roll numbers
               - Extra separators "| | |" may be ignored
            
            3. **Cross-reference within same row**:
               - If a row has "Subject: DBM__" and "Code: CS301 (DBMS)",
                 you MAY infer the subject is "DBMS" based on the code column.
               
            4. **Time format standardization**:
               - "10.00 AM" or "10:OO AM" → "10:00 AM"
               - Missing AM/PM: infer from other exams in the same table
            
            5. **Missing but inferable data**:
               - If headers show "Date | Time | Subject" but a row has "15-01-2026 | | DBMS"
                 AND other rows show morning (10 AM) and afternoon (2 PM) exams,
                 you MAY state "Time not explicitly shown, but document shows exams 
                 are typically at 10:00 AM or 2:00 PM"

            FORBIDDEN behaviors (strict anti-hallucination):
            - Do NOT invent dates with no contextual basis
            - Do NOT assume information from outside knowledge (e.g., standard exam timings)
            - Do NOT combine information from DIFFERENT DOCUMENTS
            - Do NOT make up subject names not present in any form
            - When uncertain, ALWAYS state the limitation instead of guessing

            Data normalization rules:
            - Normalize equivalent date formats (DD-MM-YYYY, DD/MM/YYYY, YYYY-MM-DD)
            - Treat numeric and date ranges as inclusive
            - Recognize that OCR may create spacing/separator artifacts

            OUTPUT FORMATTING RULES:
            Your response must be well-structured and easy to read:
            
            1. For exam schedules:
               - Start with the main answer (date and time)
               - Use clear formatting: "📅 Date: [date]" and "🕐 Time: [time]"
               - If data was inferred from context, briefly note it: "(inferred from pattern)"
            
            2. For lists or multiple items:
               - Use bullet points (•) for each item
               - Keep each point concise
            
            3. For unavailable or uncertain information:
               - Be honest: "❌ The exact [detail] is not clearly shown in the document."
               - If partial info exists: "Partial information: [what exists]"
            
            4. General structure:
               - Be concise and direct
               - Use emojis sparingly (✓, ✗, 📅, 🕐, 📄)
               - Present dates in readable format: "15th January 2026"
            
            Example good response (with inference):
            "📅 Date: 15th January 2026
            🕐 Time: 10:00 AM (inferred from document pattern - other exams at 10 AM or 2 PM)
            📄 Subject: DBMS Lab (Practical)"
            
            Example for unavailable:
            "❌ The exact time for the DBMS exam is not shown in the document.
            However, the document shows other exams are scheduled at 10:00 AM or 2:00 PM."

            Be precise. Use context wisely. Acknowledge uncertainty when appropriate.'''
    )

    user_prompt = (
        f"QUESTION:\n{query}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "FINAL ANSWER:"
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )

    return completion.choices[0].message.content.strip()


def generate_reasoning(query: str, entities: Dict[str, Any], chunks: List[Dict]) -> str:
    """
    Generate reasoning/thinking process for debugging and transparency.
    Explains what entities were found and why certain sources were selected.
    """
    # Build reasoning prompt
    entities_summary = json.dumps({k: v for k, v in entities.items() if v and not k.startswith('_')}, indent=2)
    
    sources_summary = []
    for i, chunk in enumerate(chunks[:3], 1):  # Top 3 sources for reasoning
        sources_summary.append(
            f"{i}. Page {chunk.get('page', '?')} - {chunk.get('chunk_type', 'unknown')} "
            f"(relevance: {chunk.get('distance', 0):.3f})"
        )
    
    reasoning_prompt = f"""Analyze this query and explain your thinking briefly:

Query: "{query}"

Entities Extracted:
{entities_summary if entities_summary != "{}" else "No specific entities found"}

Top Sources Retrieved:
{chr(10).join(sources_summary) if sources_summary else "No sources available"}

Provide a brief reasoning (2-3 sentences) about:
1. What key information was extracted from the query
2. How you'll use the sources to answer

Keep it concise and factual."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful reasoning assistant. Be brief and clear."},
                {"role": "user", "content": reasoning_prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Reasoning unavailable: {str(e)}"



# -------------------------------------------------------
# FULL QA PIPELINE
# -------------------------------------------------------

def extract_entities_llm(query: str) -> Dict[str, Any]:
    """
    Uses LLM to extract structured entities (roll_no, date, subject, etc.) from the query.
    """
    system_prompt = (
        """You are a query understanding engine for an academic RAG system.
        Analyze the user's question and extract filtering criteria.
        
        Output JSON only with this structure:
        {
            "intent": "examination schedule" | "fees information" | "student details" | "grades / results" | null,
            "roll_no": number | null,
            "subject": string | null,
            "date": string (DD-MM-YYYY) | null,
            "year": string (e.g. "1st", "3rd", "I", "III") | null,
            "exam_type": "regular" | "reappear" | "external" | "internal" | null
        }
        
        Rules:
        - "intent": Infer the probable table type.
          - "when is exam", "datesheet" -> "examination schedule"
          - "how much fee", "receipt" -> "fees information"
          - "my marks", "result" -> "grades / results"
          
        - "year": Extract academic year if mentioned (e.g. "3rd year" -> "3rd").
        """
    )

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(completion.choices[0].message.content)
        
        # Post-processing to match Answer.py's expected keys for row filtering
        entities = {}
        
        # 1. Roll No
        if data.get("roll_no"):
            entities["roll_no"] = str(data["roll_no"])
            try:
                entities["roll_no_val"] = float(data["roll_no"])
            except:
                pass
                
        # 2. Date
        if data.get("date"):
            entities["date"] = data["date"]
            parsed = parse_date(data["date"])
            if parsed:
                entities["date_val"] = parsed
                
        # 3. Year
        if data.get("year"):
            entities["year"] = data["year"]
            parsed_y = parse_year(data["year"])
            if parsed_y:
                entities["year_val"] = parsed_y
                
        # 4. Subject
        if data.get("subject"):
            entities["subject"] = data["subject"]
            
        # 5. Intent (For Vector DB Filtering)
        if data.get("intent"):
            entities["_intent"] = data["intent"]
            
        return entities
        
    except Exception as e:
        print(f"LLM Extraction Failed: {e}. Falling back to empty.")
        return {}


def answer_query(
    pdf_path: str,
    query: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    End-to-end QA pipeline.
    Returns: {
        "answer": str,
        "reasoning": str,
        "entities": dict,
        "sources": list
    }
    """
    # 1. Parse entities (LLM)
    entities = extract_entities_llm(query)
    
    # Construct Metadata Filters for Vector DB
    filters = {}
    # NOTE: Intent filtering removed since we removed intent from chunks
    # if "_intent" in entities:
    #     filters["intent"] = entities["_intent"]
        
    # 2. Retrieve relevant chunks (with filters)
    results = run_rag_pipeline(pdf_path, query, top_k, filters=filters)

    if not results:
        return {
            "answer": "No relevant information found in the document.",
            "reasoning": "No matching chunks were retrieved from the vector database.",
            "entities": entities,
            "sources": []
        }

    # 3. Generate reasoning
    reasoning = generate_reasoning(query, entities, results)
    
    # 4. Build context (with filtering)
    context = build_context(results, entities=entities)
    
    # Context might be empty if all tables were filtered out
    if not context.strip():
        print("Warning: Context empty after row filtering.")
        return {
            "answer": "No relevant information found (all data filtered out based on query entities).",
            "reasoning": reasoning,
            "entities": entities,
            "sources": []
        }

    # 5. Generate answer
    answer = generate_answer(query, context)
    
    # 6. Format sources
    sources = [{
        "document": pdf_path.split("/")[-1],
        "page": r.get("page", 0),
        "chunk_type": r.get("chunk_type", "unknown"),
        "relevance_score": f"{r.get('distance', 0):.3f}"
    } for r in results]

    return {
        "answer": answer,
        "reasoning": reasoning,
        "entities": entities,
        "sources": sources
    }


# -------------------------------------------------------
# TEST ENTRY
# -------------------------------------------------------

if __name__ == "__main__":
    PDF_PATH = "data/Notice 3rd year External Exam.pdf"
    QUERY = "When is the DBMS lab practical exam?"

    print("\nQUESTION:")
    print(QUERY)

    print("\nANSWER:")
    result = answer_query(PDF_PATH, QUERY)
    print(result["answer"])
    
    print("\nREASONING:")
    print(result["reasoning"])

