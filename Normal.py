"""
Normal.py: Handles text and table normalization (dates, ranges, numbers, OCR fixes).
Ensures standardized data format across the pipeline.
"""

import re
from typing import List, Dict, Any

def normalize_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return str(text) if text else ""
    
    original = text
    
    # OCR character fixes
    ocr_fixes = {
        "Il": "II", "lI": "II",
        "—": "-", "–": "-", "―": "-",
        "™": "", "®": "", "©": "(c)",
        "“": '"', "”": '"', "‘": "'", "’": "'", "„": '"', "‚": "'",
        "…": "...", "•": "*", "·": "*", "": "",
    }
    
    for wrong, correct in ocr_fixes.items():
        text = text.replace(wrong, correct)

    # Context-aware fixes
    # Fix IO- at start of string or after space (IO-123 -> 10-123)
    text = re.sub(r'\bIO-', '10-', text)
    text = re.sub(r'\bI0-', '10-', text)
    text = re.sub(r'\bl0-', '10-', text)
    
    # 2. Fix date separators (standardize to hyphen)
    # Pattern: DD.MM.YYYY or DD/MM/YYYY -> DD-MM-YYYY
    text = re.sub(r'(\d{1,2})[./](\d{1,2})[./](\d{2,4})', r'\1-\2-\3', text)
    
    # 3. Clean up multiple punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{4,}', '...', text)
    
    # 4. Fix spacing around punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)  # Ensure single space after punctuation
    
    # 5. Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # 6. Remove leading/trailing
    text = text.strip()
    text = text.strip('|')
    
    # 7. Fix number formatting
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    
    return text


def normalize_date(date_str: str) -> str:
    """
    Normalize date strings to DD-MM-YYYY format.
    Handles: DD.MM.YY, DD.MM.YYYY, DD/MM/YY, DD/MM/YYYY, YYYY-MM-DD, etc.
    Converts 2-digit years to 4-digit (assumes 20XX for YY < 50, 19XX for YY >= 50).
    """
    if not date_str:
        return date_str
    
    date_str = date_str.strip()
    
    # Already normalized (DD-MM-YYYY)
    if re.match(r'^\d{1,2}-\d{1,2}-\d{4}$', date_str):
        return date_str
    
    # DD.MM.YY or DD/MM/YY (2-digit year)
    match = re.match(r'^(\d{1,2})[./](\d{1,2})[./](\d{2})$', date_str)
    if match:
        day, month, year = match.groups()
        # Convert 2-digit year to 4-digit
        year_int = int(year)
        full_year = f"20{year}" if year_int < 50 else f"19{year}"
        return f"{day.zfill(2)}-{month.zfill(2)}-{full_year}"
    
    # DD.MM.YYYY or DD/MM/YYYY (4-digit year)
    match = re.match(r'^(\d{1,2})[./](\d{1,2})[./](\d{4})$', date_str)
    if match:
        day, month, year = match.groups()
        return f"{day.zfill(2)}-{month.zfill(2)}-{year}"
    
    # YYYY-MM-DD (reverse to DD-MM-YYYY)
    match = re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', date_str)
    if match:
        year, month, day = match.groups()
        return f"{day.zfill(2)}-{month.zfill(2)}-{year}"
    
    # If no pattern matches, return as-is
    return date_str


def normalize_number(num_str: str) -> str:
    """
    Normalize number strings (remove commas, fix OCR issues).
    """
    if not num_str or not isinstance(num_str, str):
        return str(num_str)
    
    # Remove commas in numbers
    num_str = num_str.replace(',', '')
    
    # Fix common OCR issues in numbers
    num_str = num_str.replace('O', '0')  # Letter O to zero
    num_str = num_str.replace('l', '1')  # Letter l to one
    num_str = num_str.replace('I', '1')  # Letter I to one (in number context)
    
    return num_str


# -------------------------------
# TABLE-SPECIFIC NORMALIZATION
# -------------------------------

def normalize_table_cell(cell: str) -> str:
    if not cell or not isinstance(cell, str):
        return str(cell) if cell else ""
    
    cell = cell.strip()
    
    # 1. Date Ranges (Hyphen, 'to')
    date_range_pattern = r'(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\s*[-–—to]+\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})'
    match = re.search(date_range_pattern, cell, re.IGNORECASE)
    if match:
        date1 = normalize_date(match.group(1))
        date2 = normalize_date(match.group(2))
        return f"{date1} - {date2}"
    
    # 2. Single Date
    single_date_pattern = r'^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$'
    if re.match(single_date_pattern, cell):
        return normalize_date(cell)
    
    # 3. Explicit Number Range ("100-200", "1 to 2")
    number_range_pattern = r'^(\d+)\s*[-–—to]+\s*(\d+)$'
    match = re.match(number_range_pattern, cell, re.IGNORECASE)
    if match:
        num1 = normalize_number(match.group(1))
        num2 = normalize_number(match.group(2))
        return f"{num1}-{num2}"
        
    # 4. Implicit Roll No Range ("230001 230050")
    implicit_range_pattern = r'^(\d{5,})\s+(\d{5,})$'
    match = re.match(implicit_range_pattern, cell)
    if match:
        num1 = normalize_number(match.group(1))
        num2 = normalize_number(match.group(2))
        return f"{num1}-{num2}"
    
    # 5. Is pure Number?
    if re.match(r'^\d+$', cell):
        return normalize_number(cell)
    
    # 6. Fallback Text
    return normalize_text(cell)


def normalize_table(table: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced table normalization with cell-type detection.
    Handles dates, ranges, numbers, and text appropriately.
    """
    rows = table.get("rows", [])
    norm_rows = []

    for row_idx, row in enumerate(rows):
        norm_row = []
        for cell_idx, cell in enumerate(row):
            # Use smart cell normalization
            normalized_cell = normalize_table_cell(str(cell) if cell is not None else "")
            norm_row.append(normalized_cell)
        norm_rows.append(norm_row)

    table["rows"] = norm_rows
    return table


# -------------------------------
# CHUNK NORMALIZATION
# -------------------------------

def normalize_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Normalize all text fields in chunks.
    """
    normalized = []

    for chunk in chunks:
        c = chunk.copy()

        # Normalize text fields
        if "content" in c and c["content"]:
            c["content"] = normalize_text(c["content"])

        if "embedding_text" in c and c["embedding_text"]:
            c["embedding_text"] = normalize_text(c["embedding_text"])

        if "sections" in c:
            c["sections"] = [normalize_text(s) if s else "" for s in c["sections"]]

        # Normalize tables
        if c.get("chunk_type") == "table_generic" and "table" in c:
            c["table"] = normalize_table(c["table"])

        normalized.append(c)

    return normalized


# -------------------------------
# PIPELINE ENTRY
# -------------------------------

def run_normal_pipeline(chunks: List[Dict]) -> List[Dict]:
    """
    Pure normalization layer.
    No logic. No filtering. No inference.
    Just comprehensive text cleaning and standardization.
    """
    return normalize_chunks(chunks)


# -------------------------------
# TESTING / VERIFICATION
# -------------------------------

def test_normalization():
    """
    Test normalization with common OCR issues.
    """
    print("\n=== General Text Normalization Test ===\n")
    
    text_cases = [
        "IO-11-2024",  # Should become "10-11-2024"
        "Ref.  No:  ABC/123",  # Should clean multiple spaces
        "Date: 15.01.2026",  # Should normalize to "15-01-2026"
        "Amount:  1,234,567",  # Should remove comma
        "Subject:   Web  Technology",  # Should fix spacing
        '"Smart Quotes"',  # Should normalize quotes
        "Test—document",  # Should fix em dash
        "Il year",  # Should fix to "II year"
        "Roll: 230122I720001",  # Should fix I to 1
    ]
    
    for original in text_cases:
        normalized = normalize_text(original)
        if original != normalized:
            print(f"✓ FIXED:")
            print(f"  Before: {original}")
            print(f"  After:  {normalized}\n")
        else:
            print(f"○ NO CHANGE: {original}\n")
    
    print("\n=== Table Cell Normalization Test ===\n")
    
    table_cell_cases = [
        ("15.01.26", "Date (YY format)"),
        ("15.01.2026", "Date (YYYY format)"),
        ("15-01-2026", "Already normalized date"),
        ("15.01.26 - 20.01.26", "Date range"),
        ("15/01/26 to 20/01/26", "Date range with 'to'"),
        ("100-200", "Number range"),
        ("1000 to 2000", "Number range with 'to'"),
        ("230122I720001", "Roll number with OCR error"),
        ("230001 230050", "Implicit Roll No Range (Space separated)"),
        ("BCS-552", "Course code"),
        ("Web  Technology", "Text with extra spaces"),
        ("IO bound process", "Text with IO (should stay IO)"),
        ("IO-12345", "IO prefix (should become 10)"),
    ]
    
    for original, description in table_cell_cases:
        normalized = normalize_table_cell(original)
        if original != normalized:
            print(f"✓ FIXED ({description}):")
            print(f"  Before: {original}")
            print(f"  After:  {normalized}\n")
        else:
            print(f"○ NO CHANGE ({description}): {original}\n")


def test_table_normalization():
    """
    Test full table normalization.
    """
    print("\n=== Full Table Normalization Test ===\n")
    
    test_table = {
        "rows": [
            ["Roll No", "Date", "Subject", "Code"],
            ["230122I72000I", "15.01.26", "Web  Technology", "BCS-552"],
            ["230122172OO37", "20.01.26 - 25.01.26", "Database  Management", "BCS-503"],
        ]
    }
    
    print("Before normalization:")
    for row in test_table["rows"]:
        print("  ", " | ".join(row))
    
    normalized_table = normalize_table(test_table)
    
    print("\nAfter normalization:")
    for row in normalized_table["rows"]:
        print("  ", " | ".join(row))
    
    print()


if __name__ == "__main__":
    test_normalization()
    test_table_normalization()
