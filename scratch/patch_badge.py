import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

target = r'<span className="text-xs bg-\[\#0f172a\] text-blue-400 px-3 py-1.5 rounded-full border \s*border-blue-500/30 font-medium">\s*Model: \{results\.telemetry\.model_used\}\s*</span>'

replacement = """<div className="flex items-center gap-3">
                          <div className="flex gap-3 text-[10px] text-slate-500 font-mono tracking-wide mt-1">
                             <span>{(results.telemetry?.latency_seconds || 0).toFixed(2)}s</span>
                             <span>{results.telemetry?.total_tokens || 0} tokens</span>
                          </div>
                          <span className={`text-xs px-3 py-1.5 rounded-full border font-medium flex items-center gap-2 ${results.telemetry?.model_used?.includes('Mock') ? 'bg-amber-500/10 text-amber-400 border-amber-500/30' : 'bg-blue-500/10 text-blue-400 border-blue-500/30'}`}>
                            <div className={`w-1.5 h-1.5 rounded-full ${results.telemetry?.model_used?.includes('Mock') ? 'bg-amber-500' : 'bg-emerald-500 animate-pulse'}`} />
                            {results.telemetry?.model_used?.includes('Mock') ? 'API Offline (Local Fallback)' : 'Live API: ' + results.telemetry?.model_used}
                          </span>
                        </div>"""

new_content = re.sub(target, replacement.replace('\\', '\\\\'), content)

with open("frontend/src/app/page.tsx", "w") as f:
    f.write(new_content)

print("Badge Patched successfully")
