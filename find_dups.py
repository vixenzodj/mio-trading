import re

with open('app.py', 'r') as f:
    content = f.read()

funcs = re.findall(r'^\s*def\s+([a-zA-Z0-9_]+)', content, re.MULTILINE)
counts = {}
for func in funcs:
    counts[func] = counts.get(func, 0) + 1

for func, count in counts.items():
    if count > 1:
        print(f"{func}: {count}")
