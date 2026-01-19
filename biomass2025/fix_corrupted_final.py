#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

# Read file
with open('dashboard_streamlit_app.py', 'rb') as f:
    content = f.read()

# Decode to string
text = content.decode('utf-8')

# Find and replace all corrupted patterns
# Pattern 1: Line with "Quick Stats" that has corrupted emoji
if 'Quick Stats' in text:
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'Quick Stats' in line and '###' in line:
            # Replace with correct emoji
            if '📊 Quick Stats' not in line:
                lines[i] = '    st.sidebar.markdown("### 📊 Quick Stats")'
                print(f"Line {i+1}: Fixed Quick Stats")
    
    text = '\n'.join(lines)

# Pattern 2: Data Overview line
if 'Data Overview' in text and 'elif selected_section' in text:
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'Data Overview' in line and 'elif selected_section' in line:
            # Should be: elif selected_section == "📊 Data Overview":
            if '📊 Data Overview' in line:
                continue  # Already fixed
            lines[i] = '    elif selected_section == "📊 Data Overview":'
            print(f"Line {i+1}: Fixed Data Overview check")
    
    text = '\n'.join(lines)

# Write back
with open('dashboard_streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("✓ All corrupted emojis fixed!")
