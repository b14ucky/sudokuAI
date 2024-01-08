import os


def rename_files_in_directory(directory_path, prefix):
    files = os.listdir(directory_path)

    for file_name in files:
        if file_name.startswith(prefix):
            new_file_name = "i" + file_name[len(prefix) :]

            old_path = os.path.join(directory_path, file_name)
            new_path = os.path.join(directory_path, new_file_name)

            os.rename(old_path, new_path)

            print(f"Renamed: {file_name} to {new_file_name}")


directory_path = "./numbers"
prefix_to_rename = "n"

rename_files_in_directory(directory_path, prefix_to_rename)
