#!/bin/bash
echo "Running submission validator..."
python validate_submission.py submission.csv
if [ $? -eq 0 ]; then
    echo "Validation successful!"
    echo "================ Final Submission Stats ================"
    echo "Top 5 Candidates:"
    head -n 6 submission.csv
    echo "..."
    echo "Bottom 5 Candidates:"
    tail -n 6 submission.csv
    echo "========================================================"
else
    echo "Validation failed! Please check the errors above."
fi
