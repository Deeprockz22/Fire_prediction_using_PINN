import sys

with open('fire_predict.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_str = '    print("  \\033[90mBecause running CFD in real-time is for people who enjoy waiting.\\033[0m\\n")'
new_str = '    print("  \\033[90mBecause running CFD in real-time is for people who enjoy waiting.\\033[0m\\n")\n    print("  \\033[93mThis AI has ingested the knowledge of 221 fires and 3 physics books.\\033[0m")\n    print("  \\033[93mIt is now mildly concerned about your smoking habits.\\033[0m\\n")'

if old_str in content:
    new_content = content.replace(old_str, new_str)
    with open('fire_predict.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("SUCCESS: Python patched")
else:
    print("WARNING: String not found")
