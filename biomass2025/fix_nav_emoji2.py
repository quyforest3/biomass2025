#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Read file as bytes to handle encoding properly
with open('dashboard_streamlit_app.py', 'rb') as f:
    content = f.read()

# Decode
text = content.decode('utf-8')

# Find all lines and fix the corrupted Navigation line
lines = text.split('\n')
for i, line in enumerate(lines):
    if 'Navigation' in line and '###' in line:
        # Check if it has corrupted emoji
        if '###' in line and 'Navigation' in line:
            # Fix it by replacing the entire line
            lines[i] = '    st.sidebar.markdown("### ⌨️ Navigation")'
            print(f"Fixed line {i+1}: {lines[i]}")

# Write back
new_text = '\n'.join(lines)
with open('dashboard_streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(new_text)

print("✓ Fixed Navigation emoji corruption")
