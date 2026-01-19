#!/usr/bin/env python
# -*- coding: utf-8 -*-

with open('dashboard_streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix corrupted emoji references (remove the "🗺️ AGB Prediction Map" header)
# Lines 2295 onwards have the old Prediction Map content that's now under Spatial Analysis section

# Read file by lines
with open('dashboard_streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Skip the line with the old header
    if 'st.markdown(\'<h1 class="section-header">🗺️ AGB Prediction Map</h1>\'' in line:
        continue
    # Skip lines with broken emoji encoding
    elif '️' in line and ('Prediction' in line or 'Data Overview' in line):
        # Fix the broken emoji encoding
        fixed_line = line.replace('️', '📊').replace('Prediction Map', 'Data Overview')
        new_lines.append(fixed_line)
    else:
        new_lines.append(line)

with open('dashboard_streamlit_app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✓ Cleaned up remaining Prediction Map section header")
