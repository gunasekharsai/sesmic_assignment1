# Open the .fsp file in read mode
file_path = "Downloads/srcmod2024-12-02FSP/s1906SANFRA01SONG.fsp"

try:
    with open(file_path, "r") as fsp_file:
        # Read the entire content of the file
        content = fsp_file.read()
        print("File Content:")
        print(content)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
