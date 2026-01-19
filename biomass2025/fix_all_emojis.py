#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

with open('dashboard_streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define replacements for corrupted emojis
replacements = [
    # Line 1554: Quick Stats
    ('st.sidebar.markdown("### Quick Stats")', 'st.sidebar.markdown("### 📊 Quick Stats")'),
    
    # Line 1637: Data Overview check
    ('elif selected_section == "📊 Data Overview":', 'elif selected_section == "📊 Data Overview":'),
]

original_content = content

# Apply replacements
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        print(f"✓ Fixed: {old[:50]}...")

# Also fix any remaining corrupted pattern
# Look for "### " followed by broken character
content = re.sub(
    r'st\.sidebar\.markdown\("### [^\w\s📊⚙️⭐📉🔗⚡🔬🗺️📋ℹ️🤖⌨️🛰️💡🎯🏆📈🚀💾🔧🧠📚✨👍]+\s+',
    lambda m: m.group(0).replace(m.group(0)[m.group(0).find('"### ')+5:m.group(0).find('"### ')+10], '📊 '),
    content
)

# Write back
if content != original_content:
    with open('dashboard_streamlit_app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("\n✓ All corrupted emojis have been fixed!")
else:
    print("\nℹ️ No changes needed - file appears to be clean or corrupted characters are different")
