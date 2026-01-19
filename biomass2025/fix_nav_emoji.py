#!/usr/bin/env python
# -*- coding: utf-8 -*-

with open('dashboard_streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the corrupted Navigation emoji on line 1581
content = content.replace('st.sidebar.markdown("### Navigation")', 'st.sidebar.markdown("### ⌨️ Navigation")')

with open('dashboard_streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Fixed corrupted Navigation emoji")
