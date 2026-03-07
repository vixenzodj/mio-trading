import sys

with open('app.py', 'r') as f:
    lines = f.readlines()

# Find the start of the functions
start_idx = -1
for i, line in enumerate(lines):
    if "def normalize_key(d, possible_keys):" in line:
        start_idx = i
        break

# Find the end of fetch_data_smart
end_idx = -1
for i in range(start_idx, len(lines)):
    if "return df" in lines[i] and "def fetch_data_smart" in "".join(lines[start_idx:i]):
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    # Extract the functions
    functions_code = lines[start_idx:end_idx+1]
    
    # Remove them from the original location
    del lines[start_idx:end_idx+1]
    
    # Unindent the functions by 4 spaces
    unindented_functions = []
    for line in functions_code:
        if line.startswith("    "):
            unindented_functions.append(line[4:])
        else:
            unindented_functions.append(line)
            
    # Find where to insert them (before elif menu == "🔙 BACKTESTING STRATEGIA":)
    insert_idx = -1
    for i, line in enumerate(lines):
        if 'elif menu == "🔙 BACKTESTING STRATEGIA":' in line:
            insert_idx = i
            break
            
    if insert_idx != -1:
        lines = lines[:insert_idx] + unindented_functions + ["\n"] + lines[insert_idx:]
        
        with open('app.py', 'w') as f:
            f.writelines(lines)
        print("Successfully moved functions.")
    else:
        print("Could not find insertion point.")
else:
    print("Could not find functions to move.")
