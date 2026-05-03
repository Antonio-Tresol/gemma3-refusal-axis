#!/bin/bash
# PreToolUse hook: check that findings/ files have evidence pinning
# Returns a system message reminder if writing to findings/ without citations
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python -c "import sys,json; print(json.load(sys.stdin).get('tool_input',{}).get('file_path',''))" 2>/dev/null)

# Only check writes to findings/ markdown files
if [[ "$FILE_PATH" == *findings/*.md ]] || [[ "$FILE_PATH" == *findings/**/*.md ]]; then
  NEW_CONTENT=$(echo "$INPUT" | python -c "import sys,json; d=json.load(sys.stdin).get('tool_input',{}); print(d.get('content','') or d.get('new_string',''))" 2>/dev/null)

  # Check if the content has any citation markers
  # Looking for: (Author, Year), [Source], arXiv:, §, "pinned to", "per", "following"
  HAS_CITATION=$(echo "$NEW_CONTENT" | grep -ciE '(arXiv:|et al|§[0-9]|\[[A-Z][a-z]+|pinned to|following .* 20[0-9]{2}|— \[)' 2>/dev/null)

  # Only warn on substantial content (>200 chars) without citations
  CONTENT_LEN=${#NEW_CONTENT}
  if [[ $CONTENT_LEN -gt 200 ]] && [[ "$HAS_CITATION" -eq 0 ]]; then
    echo '{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":"EVIDENCE PINNING REMINDER: This findings/ file has no visible citations or source references. Every methodological decision must cite its source. Format: [Decision] — [Source, Section/Page]. If this is a data table or code output, this warning can be ignored."}}'
    exit 0
  fi
fi

exit 0
