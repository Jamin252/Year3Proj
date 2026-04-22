pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
CODE=$?
echo "Exit Code: $CODE"
grep -E "^!|Error:|Fatal error" main.log || true
exit "$CODE"
