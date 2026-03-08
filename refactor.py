import os

with open('app.py', 'r') as f:
    lines = f.readlines()

block1_start = 883
block1_end = 1280

block2_start = 3809
block2_end = 4104

# Extract block 1
block1 = lines[block1_start:block1_end]

# Unindent block 1
unindented_block1 = []
for line in block1:
    if line.startswith('    '):
        unindented_block1.append(line[4:])
    else:
        unindented_block1.append(line)

# Remove block 2
del lines[block2_start:block2_end]

# Remove block 1
del lines[block1_start:block1_end]

# Find a good place to insert unindented_block1
# Let's put it right after fetch_alpaca_history
insert_idx = -1
for i, line in enumerate(lines):
    if "def fetch_alpaca_history" in line:
        for j in range(i+1, len(lines)):
            if not lines[j].startswith(' ') and lines[j].strip() != '':
                insert_idx = j
                break
        break

lines = lines[:insert_idx] + unindented_block1 + lines[insert_idx:]

with open('app.py', 'w') as f:
    f.writelines(lines)

print('Refactored app.py')
