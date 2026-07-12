import os
from pathlib import Path

def generate_project_context():
    # Directories and files to include
    include_dirs = ['app', 'tests', '.github']
    include_files = ['Dockerfile', 'docker-compose.yml', 'fly.toml', 'requirements.txt', 'main.py']
    
    # Extensions to include
    valid_extensions = {'.py', '.yml', '.yaml', '.toml', '.md', '.txt'}
    
    output_file = Path('project_context.txt')
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("# Zoom-to-Text Project Context\n\n")
        
        # 1. Add specific root files
        for filename in include_files:
            filepath = Path(filename)
            if filepath.exists():
                outfile.write(f"\n## File: {filename}\n```\n")
                outfile.write(filepath.read_text(encoding='utf-8', errors='replace'))
                outfile.write("\n```\n")
        
        # 2. Add directories
        for d in include_dirs:
            dir_path = Path(d)
            if not dir_path.exists():
                continue
                
            for root, _, files in os.walk(dir_path):
                for file in files:
                    filepath = Path(root) / file
                    if filepath.suffix in valid_extensions:
                        outfile.write(f"\n## File: {filepath}\n```python\n")
                        outfile.write(filepath.read_text(encoding='utf-8', errors='replace'))
                        outfile.write("\n```\n")

    print(f"[V] Project context exported to {output_file.absolute()}")

if __name__ == "__main__":
    generate_project_context()