echo "--- BibTeX Errors/Warnings ---"
grep -i "error" main.blg
grep -i "warning" main.blg
echo "--- LaTeX Undefined Citations/References ---"
grep -i "undefined" main.log
echo "--- LaTeX Errors ---"
grep -P "^! " main.log
