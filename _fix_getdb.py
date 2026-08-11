# Fix _get_db() -> state._get_db() in tasks.py
path = 'app/repositories/tasks.py'
with open(path, 'r') as f:
    content = f.read()

old1 = '''async def update_task(task_id: str, status: TaskStatus, progress: int = 0, message: str = ""):
    async with get_write_lock():
        db = await _get_db()'''

new1 = '''async def update_task(task_id: str, status: TaskStatus, progress: int = 0, message: str = ""):
    async with get_write_lock():
        db = await state._get_db()'''

old2 = '''async def append_partial_transcript(task_id: str, text: str):
    async with get_write_lock():
        db = await _get_db()'''

new2 = '''async def append_partial_transcript(task_id: str, text: str):
    async with get_write_lock():
        db = await state._get_db()'''

if old1 in content:
    content = content.replace(old1, new1)
    print("Fixed update_task")
else:
    print("WARNING: update_task pattern not found")

if old2 in content:
    content = content.replace(old2, new2)
    print("Fixed append_partial_transcript")
else:
    print("WARNING: append_partial_transcript pattern not found")

with open(path, 'w') as f:
    f.write(content)

print("Done!")
