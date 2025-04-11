#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <folder_path> <command> <pattern>"
    exit 1
fi

folder_path="$1"
command="$2"
pattern="$3"
# escape seperator before using sed
pattern=$(echo "${pattern//\//\\/}")
echo $pattern

# Check the value of the variable
if [ "$command" == "comment" ]; then
    echo "copybara comment"
    find "$folder_path" -type f -name "BUILD" | while read -r file
    do
        sed -i -e "s/$pattern/# copybara:uncomment $pattern/g" $file
    done

elif [ "$command" == "uncomment" ]; then
    echo "copybara uncomment"

    find "$folder_path" -type f -name "BUILD" | while read -r file
    do
        sed -i -e "s/# copybara:uncomment $pattern/$pattern/g" $file
    done

# TODO: handle copybara:uncomment_begin and copybara:uncomment_end
# elif [ "$command" == "comment_begin" ]; then
# elif [ "$command" == "uncomment_begin" ]; then

else
    echo "Error: The command must be either 'comment' or 'uncomment'."
    exit 1
fi
