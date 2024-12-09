from datasets import load_dataset

# Step 1: Load the dataset
dataset = load_dataset("Skylion007/openwebtext", trust_remote_code=True)

# Step 2: Specify the local directory where you want to save the dataset
local_directory = "../data/"

# Step 3: Save the dataset to the specified directory
dataset.save_to_disk(local_directory)

print(f"Dataset saved to {local_directory}")