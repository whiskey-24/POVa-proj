#!/bin/bash

# Specify the folder path
folder_path="/home/whiskey/Documents/2Mit/POVa/POVa-proj/detector/data/20181029_D1_0900_0930/20181029_D1_0900_0930/images"

# Go to the folder
cd "$folder_path" || exit

# Define the regular expression for matching filenames
pattern="[0-9][0-9][0-9][0-9][0-9]_[0-9]\.(csv|jpg)"

# List files matching the pattern and delete them
for file in $(ls | grep -E "$pattern"); do
  rm -f "$file"
  echo "Deleted: $file"
done

echo "Deletion complete."
