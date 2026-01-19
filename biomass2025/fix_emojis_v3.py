#!/usr/bin/env python
# -*- coding: utf-8 -*-

with open('dashboard_streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_count = 0

# Line 1554: Quick Stats (approximately)
for i, line in enumerate(lines):
    original_line = line
    
    # Fix "Quick Stats" line (line 1554 should be around index 1553)
    if 'Quick Stats' in line and '###' in line:
        if '📊' not in line:
            lines[i] = line.replace('### ', '### 📊 ')
            fixed_count += 1
            print(f"Line {i+1}: Fixed Quick Stats emoji")
    
    # Fix "Data Overview" check line (line 1637)
    if 'Data Overview' in line and 'elif selected_section' in line:
        if '📊' not in line or ('📊 Data Overview' not in line):
            lines[i] = line.replace('Data Overview', '📊 Data Overview').replace('📊 📊', '📊')
            fixed_count += 1
            print(f"Line {i+1}: Fixed Data Overview emoji")
    
    # Fix any line with broken emoji followed by text
    if '###' in line and line.count('###') == 1:
        # Check if there's a broken character after ###
        try:
            parts = line.split('###')
            if len(parts) >= 2:
                after_hash = parts[1][:20]
                # If first char after ### is not an emoji or space, it's broken
                if len(after_hash) > 0 and not after_hash[0].isspace() and after_hash[0] not in '📊⚙️⭐📉🔗⚡🔬🗺️📋ℹ️🤖⌨️🛰️💡🎯🏆📈🚀💾🔧🧠📚✨':
                    # This line has a corrupted character
                    print(f"Line {i+1}: Detected corrupted markdown line: {line[:60]}")
        except:
            pass

with open('dashboard_streamlit_app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print(f"\n✓ Fixed {fixed_count} corrupted emojis")
