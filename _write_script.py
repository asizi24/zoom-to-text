# Helper to write dry_run_upload.py with proper encoding
bs = chr(92)  # backslash character
content = f"""{{'''Minimal auth-free upload test.{{'}}
import requests

API = "http://localhost:8000"
VIDEO = r"C:{bs}Users{bs}Asaf{bs}Videos{bs}DevOps For Beginners{bs}Episode 2 - Linux Essential{bs}37 - Resources In Linux System.mp4"


def upload():
    filename = VIDEO.split("{bs}{bs}")[-1]
    print(f'Uploading: {{filename}}')
    with open(VIDEO, "rb") as f:
        resp = requests.post(
            f'{{API}}/api/tasks/upload',
            files={{"file": (filename, f, "video/mp4")}},
            data={{"mode": "whisper_local", "language": "he"}},
        )
    print(f'  Status: {{resp.status_code}} {{resp.text}}')
    if resp.status_code == 202:
        return resp.json().get("id")


if __name__ == "__main__":
    upload()
"""

with open(r"C:\Users\Asaf\Documents\zoom-to-text\dry_run_upload.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Written successfully")
