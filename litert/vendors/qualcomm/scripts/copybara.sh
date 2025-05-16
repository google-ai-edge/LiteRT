#!/bin/bash

print_usage() {
    echo "Usage: $0 <folder_path> <command> <pattern>"
    echo "  command: [uncomment, comment]"
    echo "  example:"
    echo "    $0 ./litert/vendors/qualcomm/ uncomment \\\"//third_party/qairt/latest:qnn_lib_headers\\\","
    echo "    $0 ./litert/vendors/qualcomm/ comment \\\"//third_party/qairt/latest:qnn_lib_headers\\\","
}

if [ "$#" -eq 1 ]; then
    if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
        print_usage
        exit 0
    fi
fi

if [ "$#" -ne 3 ]; then
    print_usage
    exit 1
fi

folder_path="$1"
command="$2"
pattern="$3"
# escape seperator before using sed
pattern=$(echo "${pattern//\//\\/}")

# Check the value of the variable
if [ "$command" == "comment" ]; then
    echo "copybara comment $pattern in $folder_path"
    find "$folder_path" -type f -name "BUILD" | while read -r file
    do
        sed -i -e "s/$pattern/# copybara:uncomment $pattern/g" $file
    done

elif [ "$command" == "uncomment" ]; then
    echo "copybara uncomment $pattern in $folder_path"

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
