import os

# Directory
BASE_DIR = os.getcwd()

# Walk through all files
for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.endswith(".html"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Replace 'img/' with 'public/images/'
            if 'src="img/' in content:
                print(f"Fixing {file}")
                new_content = content.replace('src="img/', 'src="public/images/')
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
