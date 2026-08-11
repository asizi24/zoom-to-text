content = open('app/state.py', 'r', encoding='utf-8').read()
old = 'await db.execute(CREATE_APP_CONFIG_TABLE_SQL)\n    await db.commit()'
new = '''await db.execute(CREATE_APP_CONFIG_TABLE_SQL)
    await db.execute("CREATE TABLE IF NOT EXISTS job_payload (task_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL)")
    await db.commit()'''
if old in content:
    content = content.replace(old, new)
    open('app/state.py', 'w', encoding='utf-8').write(content)
    print("SUCCESS: job_payload table injection complete")
else:
    print("ERROR: Could not find the target text to replace")
