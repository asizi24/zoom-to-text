"""Fix ollama_provider.py: replace str(parsed) with JSON-validation-only approach."""

file_path = r"app\services\llm\ollama_provider.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

old_block = '''        # Now parse the actual cleaned LLM JSON output
        try:
            parsed = json.loads(llm_text)
            return str(parsed)  # callers expect plain string/text
        except (json.JSONDecodeError, TypeError):
            # Model didn't produce valid JSON — fall back to raw text
            logger.warning("Ollama response is not valid JSON; returning raw text")
            return llm_text.strip()'''

new_block = '''        # Attempt to validate as JSON (catches truly invalid responses early)
        # Then return the cleaned text — callers expect plain string/text, not a dict repr.
        try:
            json.loads(llm_text)  # validate only — do NOT replace llm_text with parsed result
        except (json.JSONDecodeError, TypeError):
            pass  # Model generated markdown or mixed content; return as-is

        return llm_text'''

if old_block in content:
    content = content.replace(old_block, new_block)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: str(parsed) block replaced with JSON validation only")
else:
    print("ERROR: old_block not found in file")
    # Debug: show what's around line ~135
    lines = content.split('\n')
    for i, line in enumerate(lines[125:140], start=126):
        print(f"  {i}: {repr(line)}")
