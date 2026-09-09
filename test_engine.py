import ast
from pathlib import Path
p=Path(__file__).with_name('app.py')
ast.parse(p.read_text(encoding='utf-8'))
print('PASS: app.py parses successfully')
