#!/usr/bin/env python
# -*- coding: utf-8 -*-

with open('dashboard_streamlit_app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Remove lines 1606-1607 (Prediction Map button) and 1639 (Prediction Map tip)
new_lines = []
for i, line in enumerate(lines):
    line_num = i + 1
    # Skip Prediction Map button lines (1606-1607)
    if line_num in [1606, 1607]:
        continue
    # Fix line 1609 (Data Overview button) - was corrupted by PowerShell removal
    elif line_num == 1609:
        new_lines.append('    if st.sidebar.button("📊 Data Overview", use_container_width=True):\n')
    # Skip Prediction Map section check (1638-1640)
    elif line_num in [1638, 1639]:
        continue
    # Skip Prediction Map tip (1649)
    elif line_num == 1649:
        continue
    else:
        new_lines.append(line)

with open('dashboard_streamlit_app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("✓ Successfully removed Prediction Map button and tips from sidebar")
