#!/bin/bash
# Post-edit hook: runs ruff check --fix + ruff format (with isort) on .py files
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)

if [[ "$FILE_PATH" == *.py && -f "$FILE_PATH" ]]; then
  uvx ruff check --fix --select I "$FILE_PATH" 2>/dev/null
  uvx ruff format "$FILE_PATH" 2>/dev/null
fi

exit 0
