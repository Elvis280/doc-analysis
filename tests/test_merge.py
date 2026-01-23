"""
Test Vertical Merge Repair Logic
"""

def repair_table(rows):
    if not rows: return []
    
    repaired_rows = []
    # Keep header as is
    if len(rows) > 0:
        repaired_rows.append(rows[0])
    
    for i in range(1, len(rows)):
        curr_row = rows[i]
        prev_row = repaired_rows[-1] # Use previously repaired row to propagate merges
        
        new_row = []
        
        for col_idx in range(len(curr_row)):
            cell = curr_row[col_idx]
            
            # Check if cell is effectively empty
            is_empty = cell is None or str(cell).strip() == ""
            
            # Heuristic: If empty, fill from previous row
            # BUT FIRST: Check if the value we are copying is a HEADER keyword.
            candidate_value = prev_row[col_idx] if col_idx < len(prev_row) else None
            
            is_header_val = False
            if candidate_value and isinstance(candidate_value, str):
                header_keywords = {
                    "date", "day", "time", "subject", "sub", "code", "paper", 
                    "branch", "sem", "semester", "name", "roll", "no", "room", 
                    "venue", "exam", "course", "year", "session", "lab"
                }
                clean_val = str(candidate_value).strip().lower()
                clean_val = clean_val.replace('.', '').replace(':', '')
                
                if clean_val in header_keywords:
                    is_header_val = True
                elif any(k in clean_val for k in header_keywords) and len(clean_val) < 15:
                    is_header_val = True
                    
            if is_empty and candidate_value is not None and not is_header_val:
                new_row.append(candidate_value)
            else:
                new_row.append(cell)
                
        repaired_rows.append(new_row)
    
    return repaired_rows

def test_merges():
    # Test Case 1: Standard Vertical Merge
    print("Test Case 1: Standard Vertical Merge")
    rows = [
        ["Date", "Subject", "Code"],
        ["15-01-2026", "Math", "M101"],
        ["", "Physics", "P101"],      # Date should merge down
        ["", "Chemistry", "C101"],    # Date should propagate down
        ["20-01-2026", "English", "E101"]
    ]
    
    repaired = repair_table(rows)
    for r in repaired:
        print(r)
        
    print("\nTest Case 2: Multi-column Merge")
    rows2 = [
        ["Col1", "Col2", "Col3"],
        ["A", "B", "C"],
        ["", "", "D"], # Should become A, B, D
        ["", "E", ""]  # Should become A, E, D?? Note: Prev row for Col3 was D, not C. Propagates D.
                        # Wait, merge usually means "cell spans". 
                        # If row 2 col 3 is D, then row 3 col 3 being empty implies D continues? 
                        # Yes, that's consistent with "merged cell".
    ]
    
    repaired2 = repair_table(rows2)
    for r in repaired2:
        print(r)

    print("\nTest Case 3: Header Bleed Prevention")
    rows3 = [
        ["Date", "Subject"],
        ["", "Math"],        # Should NOT inherit "Date"
        ["15-01-26", "Physics"],
        ["", "Chemistry"]    # Should inherit "15-01-26"
    ]
    repaired3 = repair_table(rows3)
    for r in repaired3:
        print(r)

if __name__ == "__main__":
    test_merges()
