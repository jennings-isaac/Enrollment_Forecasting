#!/bin/bash

# Scripts to run
scripts=("create_datasets.py" "curation.py")

#iterate over each script to execute it
for script in "${scripts[@]}"; do
    python3 "$script"

    # Check if the script ran properly
    if [ $? -ne 0 ]; then
        echo "Error while running $script"
        exit 1 # Quit if error occurs
    fi
    echo "Finished $script..."
done
echo "All scripts ran successfully!"