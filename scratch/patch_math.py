import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

target = r"if \(el\) el\.innerText = '\+\$' \+ Math\.max\(0, data\.lift\)\.toLocaleString\(undefined, \{maximumFractionDigits:0\}\);"
replacement = """const val = data.lift;
                                  const prefix = val >= 0 ? '+$' : '-$';
                                  if (el) el.innerText = prefix + Math.abs(val).toLocaleString(undefined, {maximumFractionDigits:0});"""

new_content = re.sub(target, replacement, content)

with open("frontend/src/app/page.tsx", "w") as f:
    f.write(new_content)

print("Page UI Patched")
