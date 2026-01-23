"""
Clean.py: Processes OCR output into structured semantic chunks (Text & Tables).
Includes advanced table repair logic (Forward/Backward Fill, Horizontal Merge) to handle complex exam schedules.
"""

import re
import string
from Extract import run_ocr_pipeline
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_line(line):
    return re.sub(r"\s+", " ", line).strip()

def alpha_ratio(line):
    letters = sum(c.isalpha() for c in line)
    return letters / max(1, len(line))


def is_header(line):
    return (
        len(line) <= 50 and
        alpha_ratio(line) > 0.6 and
        line.upper() == line
    )


def is_meta_line(line):
    keywords = ["ref.", "date", "department", "exam"]
    l = line.lower()
    return any(k in l for k in keywords)

def is_section_label(line):
    return (
        "semester" in line.lower() and
        len(line) <= 30
    )

def is_noise(line):
    if len(line) < 6:
        return True

    # If it has 'semester' or 'ref' or digits, it's probably meaningful
    if re.search(r"(semester|ref\.|date|\d)", line.lower()):
        return False

    if alpha_ratio(line) < 0.25:
        return True

    # Too many symbols and no words
    symbols = sum(not c.isalnum() and not c.isspace() for c in line)
    return symbols / len(line) > 0.3


def is_paragraph(line):
    return (
        len(line) > 50 and
        "." in line
    )

def classify_text_content(text):
    """
    Classify text chunk into semantic types based on content.
    Returns: chunk_type string
    """
    text_lower = text.lower()
    
    # Check for rules/regulations (strong indicators)
    rule_keywords = ["must", "should", "required", "mandatory", "not allowed", 
                     "prohibited", "shall", "regulation", "guideline"]
    if any(word in text_lower for word in rule_keywords):
        return "rule"
    
    # Check for notices (official communication)
    notice_keywords = ["notice", "inform", "attention", "hereby", "notification",
                       "kindly note", "please note"]
    if any(word in text_lower for word in notice_keywords):
        return "notice"
    
    # Check for instructions (action-oriented)
    instruction_keywords = ["please submit", "fill", "complete", "follow", 
                           "procedure", "steps", "how to"]
    if any(word in text_lower for word in instruction_keywords):
        return "instruction"
    
    # Check for announcements (time-sensitive updates)
    announcement_keywords = ["announced", "postponed", "rescheduled", "declared",
                            "cancelled", "extended"]
    if any(word in text_lower for word in announcement_keywords):
        return "announcement"
    
    # Default for general text
    return "general_text"

def group_lines(lines):
    blocks = []
    buffer = []

    for line in lines:
        if is_header(line) or is_section_label(line) or is_meta_line(line):
            if buffer:
                blocks.append(" ".join(buffer))
                buffer = []
            blocks.append(line)
        else:
            buffer.append(line)

    if buffer:
        blocks.append(" ".join(buffer))

    return blocks



def detect_blocks_from_page(page_text, tables, page_num):
    lines = [clean_line(l) for l in page_text.split("\n") if l.strip()]
    lines = [l for l in lines if not is_noise(l)]

    merged = group_lines(lines)

    blocks = []

    for item in merged:
        if is_header(item):
            blocks.append({
                "type": "header",
                "text": item,
                "page": page_num
            })

        elif is_meta_line(item):
            blocks.append({
                "type": "meta",
                "text": item,
                "page": page_num
            })

        elif is_section_label(item):
            blocks.append({
                "type": "section_label",
                "text": item,
                "page": page_num
            })

        elif is_paragraph(item):
            blocks.append({
                "type": "paragraph",
                "text": item,
                "page": page_num
            })

        else:
            blocks.append({
                "type": "line",
                "text": item,
                "page": page_num
            })

    # Inject tables as first-class blocks
    for t_idx, table in enumerate(tables):
        blocks.append({
            "type": "table",
            "table_id": t_idx + 1,
            "table": table,
            "page": page_num
        })

    return blocks


# section label ko table se link krna hai

def link_tables_across_pages(pages_blocks):
    """
    pages_blocks: list of (page_num, blocks)
    returns: same structure with 'sections' list attached to tables
    """
    current_sections = []
    pending_sections = []

    for page_num, blocks in pages_blocks:
        for block in blocks:
            btype = block["type"]
            
            if btype == "section_label":
                pending_sections.append(block["text"])
            
            # Meaningful content triggers commitment
            elif btype in {"paragraph", "line", "table"}:
                if pending_sections:
                    current_sections = list(pending_sections)
                    pending_sections = []
                
                if btype == "table":
                    # If no sections active yet, use GLOBAL
                    block["sections"] = list(current_sections) if current_sections else ["GLOBAL"]
            
            # Headers and Meta do NOT trigger commitment
            # They just pass through without affecting the state
            
    return pages_blocks


def pretty_print_blocks(pages_blocks):
    for page_data in pages_blocks:
        page_num = page_data["page"]
        blocks = page_data["blocks"]

        print("\n" + "=" * 60)
        print(f"PAGE {page_num}")
        print("=" * 60)

        for b in blocks:
            if b["type"] == "header":
                print(f"\n[HEADER] {b['text']}")

            elif b["type"] == "meta":
                print(f"[META] {b['text']}")

            elif b["type"] == "section_label":
                print(f"\n[SECTION] {b['text']}")

            elif b["type"] == "paragraph":
                print(f"[PARAGRAPH] {b['text']}")

            elif b["type"] == "line":
                print(f"[LINE] {b['text']}")

            elif b["type"] == "table":
                # Join sections for display
                secs = b.get("sections", [])
                sec_str = ", ".join(secs) if secs else "UNKNOWN_SECTION"
                print(
                    f"\n[TABLE {b['table_id']}] "
                    f"(Sections: [{sec_str}])"
                )
                for row in b["table"].get("rows", []):
                    print("   ", row)

def detect_table_signals(table):
    rows = table.get("rows", [])
    if not rows:
        return {}

    lengths = [len(r) for r in rows]
    unique_lengths = set(lengths)
    
    # Check for repeated header-like rows (heuristic: first row repeats exactly)
    repeated_header = False
    if len(rows) > 5:
        first_row_str = str(rows[0])
        for r in rows[5:]: # Check further down
            if str(r) == first_row_str:
                repeated_header = True
                break
    
    return {
        "possible_header_row": True, # Process always passes header as first row currently
        "repeated_header_detected": repeated_header,
        "merged_cells_likely": any(c is None or c == "" for r in rows for c in r), # Crude heuristic
        "row_length_variance": len(unique_lengths) > 1
    }

def build_chunks(pages_blocks):
    chunks = []
    # Announcements are global across documents (or strictly distinct? "Deduplicated by normalized text")
    # We keep global deduplication for announcements as they often repeat exactly across pages.
    seen_announcements = set()

    chunk_idx = 0 # for stable IDs

    for page_data in pages_blocks:
        blocks = page_data["blocks"]
        current_page = page_data["page"]
        
        # Page-Scoped Buffers
        meta_buffer = []
        seen_meta = set()
        paragraph_buffer = []

        for block in blocks:

            # ---------- META ----------
            if block["type"] == "meta":
                # Deduplicate metadata lines LOCALLY to the page
                text = block["text"]
                norm_key = re.sub(r"\s+", " ", text).strip().lower()
                
                if norm_key and norm_key not in seen_meta:
                    meta_buffer.append(text.strip())
                    seen_meta.add(norm_key)

            # ---------- PARAGRAPH ----------
            elif block["type"] == "paragraph":
                paragraph_buffer.append(block["text"])

            # ---------- GENERIC TABLE ----------
            elif block["type"] == "table":
                t = block["table"]
                
                # Fix: Handle merged cells (vertical fill)
                # Many tables have merged cells where "Date" or "Subject" is only in first row of the group
                rows = t.get("rows", [])
                if rows:
                    repaired_rows = []
                    # Keep header as is
                    if len(rows) > 0:
                        repaired_rows.append(rows[0])
                    
                    # 1. Forward Fill (Down)
                    # 1. Forward Fill (Down)
                    for i in range(1, len(rows)):
                        curr_row = rows[i]
                        prev_row = repaired_rows[-1]
                        
                        new_row = []
                        for col_idx in range(len(curr_row)):
                            cell = curr_row[col_idx]
                            is_empty = cell is None or str(cell).strip() == ""
                            
                            # Define candidate value from previous row
                            candidate_value = prev_row[col_idx] if col_idx < len(prev_row) else None
                            
                            # Check if it's a header
                            is_header_val = False
                            is_id_val = False
                            
                            if candidate_value and isinstance(candidate_value, str):
                                # Header Parsing
                                header_keywords = {
                                    "date", "day", "time", "subject", "sub", "code", "paper", 
                                    "branch", "sem", "semester", "name", "roll", "no", "room", 
                                    "venue", "exam", "course", "year", "session", "lab", "internal", "examiner"
                                }
                                clean_val = str(candidate_value).strip().lower()
                                clean_val = clean_val.replace('.', '').replace(':', '')
                                if clean_val in header_keywords or (len(clean_val) < 15 and any(k in clean_val for k in header_keywords)):
                                    is_header_val = True
                                
                                # ID/RollNo Check (prevent duplication of single IDs)
                                v_clean = re.sub(r'[^0-9]', '', str(candidate_value))
                                if len(v_clean) > 8: # Roll numbers usually >8 digits
                                    is_id_val = True
                            
                            # Valid fill condition: Not header, Not ID
                            if is_empty and candidate_value is not None and not is_header_val and not is_id_val:
                                new_row.append(candidate_value)
                            else:
                                new_row.append(cell)
                        repaired_rows.append(new_row)
                    
                    # 2. Backward Fill (Up) - for vertically centered text
                    # We iterate backwards from end to 1 (skipping header 0)
                    for i in range(len(repaired_rows) - 2, 0, -1):
                        curr_row = repaired_rows[i]
                        next_row = repaired_rows[i+1] # Row below
                        
                        # Safety: Don't backward fill INTO a header row
                        # If current row contains any header keywords (Subject, Date, etc.), skip it
                        row_is_header = False
                        header_keywords_check = {
                            "date", "day", "time", "subject", "sub", "code", "paper", 
                            "branch", "sem", "semester", "name", "roll", "no", "room", 
                            "venue", "exam", "course", "year", "session", "lab", "internal", "examiner",
                            "students", "total", "signature"
                        }
                        for c in curr_row:
                            if c and isinstance(c, str):
                                c_clean = str(c).strip().lower().replace('.', '').replace(':', '')
                                if c_clean in header_keywords_check or (len(c_clean) < 15 and any(k in c_clean for k in header_keywords_check)):
                                    row_is_header = True
                                    break
                        
                        if row_is_header:
                            continue
                        
                        for col_idx in range(len(curr_row)):
                            cell = curr_row[col_idx]
                            is_empty = cell is None or str(cell).strip() == ""
                            next_val = next_row[col_idx] if col_idx < len(next_row) else None
                            
                            is_id_val = False
                            if next_val and isinstance(next_val, str):
                                v_clean = re.sub(r'[^0-9]', '', next_val)
                                if len(v_clean) > 8:
                                    is_id_val = True

                            # Only fill if next value is NOT an ID
                            if is_empty and next_val and not is_id_val:
                                curr_row[col_idx] = next_val

                    # 3. Horizontal Merge for Roll Number Ranges
                    # Look for pattern: [BigNumber, ..., BigNumber] in same row
                    for row in repaired_rows:
                        # Find indices of large numbers (10+ digits)
                        num_indices = []
                        for idx, cell in enumerate(row):
                            if cell and isinstance(cell, str):
                                # Clean potential non-breaking spaces
                                c_clean = cell.strip()
                                if c_clean.isdigit() and len(c_clean) >= 10:
                                    num_indices.append((idx, c_clean))
                        
                        # If we found disjoint numbers that could be a range
                        # heuristic: If we have Col A and Col C with numbers, and Col B is empty or small?
                        # Or just: if we have 2+ numbers, and they share prefix?
                        if len(num_indices) >= 2:
                            # Check first and last mostly
                            # If we have exactly 2 big numbers, or 3 where one is Start, one is End
                            # User case: Col 0 (Start), Col 2 (End). Col 1 is "2201..." (Other)
                            
                            # Let's try to form a range from first and last if they look related
                            first_idx, first_val = num_indices[0]
                            last_idx, last_val = num_indices[-1]
                            
                            if first_idx != last_idx and len(first_val) == len(last_val):
                                # Check if they form a logical range (start < end)
                                try:
                                    if int(first_val) < int(last_val):
                                        # Merge them!
                                        range_str = f"{first_val} - {last_val}"
                                        row[first_idx] = range_str
                                        # Clear the used "End" cell to avoid duplicate data (or keep it? better clear to avoid confusion)
                                        # But wait, what about the middle columns?
                                        # If we merge idx 0 and idx 2, row[2] becomes redundant?
                                        # Let's set row[last_idx] to empty or "-" to indicate merged
                                        row[last_idx] = "" 
                                except:
                                    pass

                    t["rows"] = repaired_rows
                
                rows = t.get("rows", [])
                
                # Generate Lightweight Embedding Text
                sections = block.get("sections", [])
                section_str = ", ".join(sections) if sections else "General"
                
                header_str = "Unknown"
                if rows:
                    # Fix 2: Clean Table Header Text for Embeddings
                    # Normalize, remove newlines, skip empty
                    header_cells = rows[0]
                    cleaned_headers = [
                        re.sub(r"\s+", " ", str(c)).strip() 
                        for c in header_cells 
                        if c and str(c).strip()
                    ]
                    if cleaned_headers:
                        header_str = ", ".join(cleaned_headers)
                
                data_row_count = max(0, len(rows) - 1)
                
                embedding_text = (
                    f"Table related to sections {section_str}. "
                    f"Columns include {header_str}. "
                    f"Contains {data_row_count} data rows."
                )

                chunk_id = f"{current_page}_table_{chunk_idx}"
                chunk_idx += 1

                chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_type": "table_generic",
                    "page": block["page"],
                    "sections": sections,
                    "table": t,
                    "signals": detect_table_signals(t),
                    "embedding_text": embedding_text
                })

            # ---------- FLUSH PARAGRAPHS ----------
            elif block["type"] in {"section_label", "header"}:
                if paragraph_buffer:
                    content = " ".join(paragraph_buffer).strip()
                    norm_content = re.sub(r"\s+", " ", content).lower()
                    
                    if content and norm_content not in seen_announcements:
                        # Classify chunk type dynamically
                        chunk_type = classify_text_content(content)
                        
                        # Use RecursiveCharacterTextSplitter for robust splitting
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            separators=["\n\n", "\n", ".", " ", ""]
                        )
                        splits = text_splitter.split_text(content)
                        
                        for split_idx, split_content in enumerate(splits):
                            chunk_id = f"{current_page}_{chunk_type}_{chunk_idx}_{split_idx}"
                            
                            chunks.append({
                                "chunk_id": chunk_id,
                                "chunk_type": chunk_type,
                                "page": block["page"], 
                                "content": split_content
                            })
                        
                        chunk_idx += 1
                        seen_announcements.add(norm_content)
                        
                    paragraph_buffer = []

        # Flush leftover paragraph per page
        if paragraph_buffer:
            content = " ".join(paragraph_buffer).strip()
            norm_content = re.sub(r"\s+", " ", content).lower()
            
            if content and norm_content not in seen_announcements:
                # Classify chunk type dynamically
                chunk_type = classify_text_content(content)
                
                # Use RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ".", " ", ""]
                )
                splits = text_splitter.split_text(content)
                
                for split_idx, split_content in enumerate(splits):
                    chunk_id = f"{current_page}_{chunk_type}_{chunk_idx}_{split_idx}"
                    
                    chunks.append({
                        "chunk_id": chunk_id,
                        "chunk_type": chunk_type,
                        "page": current_page,
                        "content": split_content
                    })
                
                chunk_idx += 1
                seen_announcements.add(norm_content)
            # paragraph_buffer reset automatically at loop start
            
        # Flush Page-Scoped Metadata
        if meta_buffer:
            # Metadata is usually the "first" semantic block, but we append it at end of loop logic?
            # It's better to verify if we want metadata chunks to have specific IDs.
            # We'll just append it here.
            chunk_id = f"{current_page}_metadata_{chunk_idx}"
            chunk_idx += 1
            
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_type": "metadata",
                "page": current_page,
                "content": " ".join(meta_buffer)
            })

    return chunks




def run_clean_pipeline(pdf_path):
    """
    Main entry point for document cleaning and structural chunking.
    Returns a list of generic chunks.
    """
    res = run_ocr_pipeline(pdf_path)
    if not res:
        return []

    all_pages_blocks = []
    for page in res["pages"]:
        blocks = detect_blocks_from_page(
            page_text=page["text"],
            tables=page["tables"],
            page_num=page["page_num"]
        )
        all_pages_blocks.append((page["page_num"], blocks))

    linked_pages = link_tables_across_pages(all_pages_blocks)

    structured_pages = []
    for page_num, blocks in linked_pages:
        structured_pages.append({
            "page": page_num,
            "blocks": blocks
        })

    chunks = build_chunks(structured_pages)
    return chunks

# -------------------------
# MAIN TEST ENTRY POINT
# -------------------------
if __name__ == "__main__":
    import json

    test_pdf = "data/Notice 3rd year External Exam.pdf"
    print(f"Running pipeline on {test_pdf}...")
    
    chunks = run_clean_pipeline(test_pdf)

    print("\n" + "#" * 60)
    print("GENERIC SEMANTIC CHUNKS")
    print("#" * 60)

    for i, c in enumerate(chunks, 1):
        print(f"\nChunk {i}: {c['chunk_type']} [ID: {c.get('chunk_id', '?')}]")
        if c["chunk_type"] == "table_generic":
            print(" Sections:", c["sections"])
            print(" Signals:", c["signals"])
            print(" Embedding Text:", c.get("embedding_text", ""))
            print(f" Rows (Raw): {len(c['table'].get('rows', []))}")
        else:
            print(" Content:", c.get("content", ""))


