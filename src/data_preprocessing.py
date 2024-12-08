import os

# Define the folder containing the .txt files
folder_path = '../data/text9/'  # Replace with your folder path
output_path = '../data/'
output_file = os.path.join(output_path, "input.txt")

# Open the output file in write mode
with open(output_file, "w", encoding="utf-8") as outfile:
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") and filename != "input.txt":
            file_path = os.path.join(folder_path, filename)
            # Read each file and write its content to the output file
            with open(file_path, "r", encoding="utf-8") as infile:
                outfile.write(infile.read() + "\n")  # Add a newline between files

print(f"All .txt files combined into {output_file}")
