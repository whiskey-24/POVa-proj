#!/bin/bash

# Set the paths to the source folders
yolo1_folder="data/yolo1"
yolo10_folder="data/yolo10"

# Set the path to the destination folder
yolo_folder="data/yolo"

echo "Creating destination folder '$yolo_folder'..."
# Create the destination folder and subfolders if they don't exist
mkdir -p "$yolo_folder/train/images"
mkdir -p "$yolo_folder/train/labels"
mkdir -p "$yolo_folder/test/images"
mkdir -p "$yolo_folder/test/labels"

# Function to copy and rename files
copy_and_rename() {
    src_folder=$1
    suffix=$2

    for set_folder in "train" "test"; do
      echo "Copying and renaming images with suffix '$suffix' in '$set_folder' set..."
        for file_path in "$src_folder/$set_folder/images"/*; do
            if [ -f "$file_path" ]; then
                file_name=$(basename "$file_path")
                file_extension="${file_name##*.}"
                file_name_no_ext="${file_name%.*}"

                new_file_name="$file_name_no_ext$suffix.$file_extension"
                cp "$file_path" "$yolo_folder/$set_folder/images/$new_file_name"
            fi
        done

        echo "Copying and renaming labels with suffix '$suffix' in '$set_folder' set..."
        for file_path in "$src_folder/$set_folder/labels"/*; do
            if [ -f "$file_path" ]; then
                file_name=$(basename "$file_path")
                file_extension="${file_name##*.}"
                file_name_no_ext="${file_name%.*}"

                new_file_name="$file_name_no_ext$suffix.$file_extension"
                cp "$file_path" "$yolo_folder/$set_folder/labels/$new_file_name"
            fi
        done
    done
}

echo "Copying and renaming images with suffix '_1'..."
# Copy and rename images with suffix "_1"
copy_and_rename "$yolo1_folder" "_1"

echo "Copying and renaming images with suffix '_10'..."
# Copy and rename images with suffix "_10"
copy_and_rename "$yolo10_folder" "_10"

echo "Merge completed successfully."
