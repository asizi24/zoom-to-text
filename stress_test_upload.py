"""Stress test uploader for zoom-to-text pipeline."""
import requests
import time
import sys

API = "http://localhost:8000/api"
VIDEO_PATH = r"C:\Users\Asaf\Videos\DevOps For Beginners\Episode 2 - Linux Essential\שיעור 37 -  Resources In Linux System.mp4"

def main():
    # 1. Upload file
    print(f"\n{'='*60}")
    print(f"UPLOADING: {VIDEO_PATH}")
    print(f"Size: {__import__('os').path.getsize(VIDEO_PATH) / 1e6:.1f} MB")
    print(f"{'='*60}\n")

    with open(VIDEO_PATH, 'rb') as f:
        files = {'file': ('lesson37.mp4', f, 'video/mp4')}
        data = {'mode': 'whisper_local', 'language': 'he'}
        
        resp = requests.post(f"{API}/tasks/upload", files=files, data=data)
    
    if resp.status_code not in (200, 201, 202):
        print(f"ERROR: {resp.status_code} - {resp.text}")
        sys.exit(1)
    
    task = resp.json()
    task_id = task['id']
    print(f"\n✅ Task created: {task_id}")
    print(f"   Status: {task['status']}")
    print(f"{'='*60}\n")
    
    # 2. Poll for completion
    poll_url = f"{API}/tasks/{task_id}"
    last_progress = 0
    start = time.time()
    
    while True:
        resp = requests.get(poll_url)
        if resp.status_code != 200:
            print(f"\nERROR polling: {resp.text}")
            sys.exit(1)
        
        task = resp.json()
        status = task['status']
        progress = task.get('progress', 0)
        message = task.get('message', '')
        
        elapsed = time.time() - start
        
        # Print progress change
        if progress != last_progress or status not in ('PENDING', 'COMPLETED'):
            bar_len = 30
            filled = int(bar_len * progress / 100) if progress > 0 else 0
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f"\r[{bar}] {progress:3d}% | {status:20s} | {message}", end='', flush=True)
            last_progress = progress
        
        if status in ('COMPLETED',):
            elapsed_min = elapsed / 60
            print(f"\n\n{'='*60}")
            print(f"✅ COMPLETED in {elapsed_min:.1f} minutes")
            print(f"{'='*60}\n")
            
            # Print summary if available
            result = task.get('result', {})
            if result and isinstance(result, dict):
                summary = result.get('summary', '')
                chapters = result.get('chapters', [])
                quiz = result.get('quiz', [])
                
                print(f"Summary length: {len(summary) if summary else 0} chars")
                print(f"Chapters: {len(chapters) if chapters else 0}")
                print(f"Quiz questions: {len(quiz) if quiz else 0}")
                
                # Check for RTL/LTR issues
                if summary:
                    has_hebrew = any('\u0590' <= c <= '\u05FF' for c in summary)
                    has_code_switching = any(term in summary.lower() 
                                           for term in ['linux', 'system', 'resource', 'disk'])
                    print(f"\nRTL/LTR Analysis:")
                    print(f"  Contains Hebrew: {has_hebrew}")
                    print(f"  Has code-switching (English terms): {has_code_switching}")
                    
                    # Show first 200 chars
                    display = summary[:300].replace('\n', ' ')
                    print(f"\nSummary preview:\n{display}...")
                
                if quiz and len(quiz) > 0:
                    q0 = quiz[0] if isinstance(quiz[0], dict) else {}
                    print(f"\nFirst question sample:")
                    print(f"  Question: {q0.get('question', '')[:100]}...")
                    options = q0.get('options', [])
                    if options and len(options) > 0:
                        for i, opt in enumerate(options):
                            opt_text = opt if isinstance(opt, str) else opt.get('text', str(opt))
                            print(f"    Option {i+1}: {opt_text[:80]}...")
            
            # Save full result to file for detailed review
            import json
            with open('/tmp/stress_test_result.json', 'w', encoding='utf-8') as out:
                json.dump(task, out, ensure_ascii=False, indent=2)
            print(f"\nFull result saved to: /tmp/stress_test_result.json")
            
            break
            
        elif status in ('FAILED', 'CANCELLED'):
            error = task.get('error', 'No error details')
            print(f"\n\n{'='*60}")
            print(f"❌ FAILED: {error}")
            if task.get('error_detail'):
                print(f"Detail: {task['error_detail'][:200]}")
            print(f"{'='*60}\n")
            
            # Save failure details
            with open('/tmp/stress_test_failure.json', 'w', encoding='utf-8') as out:
                json.dump(task, out, ensure_ascii=False, indent=2)
            break
        
        time.sleep(2)

if __name__ == '__main__':
    main()
