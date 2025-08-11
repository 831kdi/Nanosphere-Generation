# check_imports.py
import ast, importlib, sys
from pathlib import Path

proj_root = Path(__file__).parent.resolve()
src_root  = proj_root / "src"

# 1) Make your local 'mattersim' package importable:
if src_root.exists():
    sys.path.insert(0, str(src_root))

# 2) Gather all the .py files to scan:
files = [proj_root / "nanosphere_md.py"]
if src_root.exists():
    files += list(src_root.rglob("*.py"))

# 3) Parse out only ABSOLUTE import names:
modules = set()
for path in files:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                modules.add(n.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            # skip any relative imports (level>0)
            if node.level == 0 and node.module:
                modules.add(node.module.split(".", 1)[0])

# 4) Try to import each one:
missing = []
print("Checking imports in:")
for f in files:
    print("  ", f.relative_to(proj_root))
for m in sorted(modules):
    try:
        importlib.import_module(m)
        print(f"  ✓ {m}")
    except ImportError:
        print(f"  ✗ {m}")
        missing.append(m)

# 5) Summary
if missing:
    print("\nMissing modules:", ", ".join(missing))
    sys.exit(1)
else:
    print("\nAll imports resolved successfully!")

