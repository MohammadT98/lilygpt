import json, traceback, sys
from pathlib import Path
nb_path = Path(__file__).parents[1] / 'notebooks' / 'prompt_lab.ipynb'
print('Checking', nb_path)
nb = json.loads(nb_path.read_text(encoding='utf-8'))
errors = []
for i, cell in enumerate(nb.get('cells', []), 1):
    if cell.get('cell_type') != 'code':
        continue
    src = ''.join(cell.get('source', []))
    try:
        compile(src, f'<cell {i}>', 'exec')
    except Exception as e:
        errors.append((i, e, traceback.format_exc()))

if not errors:
    print('OK: no syntax errors detected in code cells')
    sys.exit(0)
else:
    for i, e, tb in errors:
        print('---')
        print(f'Cell {i} syntax error: {e}')
        print(tb)
    sys.exit(2)
