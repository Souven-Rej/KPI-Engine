import re

with open('frontend/src/app/page.tsx', 'r') as f:
    content = f.read()

# 1. Add state variables for Custom Mode
state_vars = """
  const [customMode, setCustomMode] = useState(false);
  const [customData, setCustomData] = useState({
    region: "West",
    ad_spend: 1500,
    web_traffic: 5000,
    stock_on_hand: 200,
    net_revenue: 8000
  });
"""
content = re.sub(
    r'(const \[error, setError\] = useState\(null\);)',
    r'\1\n' + state_vars,
    content
)

# 2. Add handleAnalyzeCustom
handle_custom = """
  const handleAnalyzeCustom = async () => {
    setLoading(true);
    setError(null);
    setResults(null);
    setHistory(null);
    try {
      const [histRes, analRes] = await Promise.all([
        fetch(`${API_BASE}/api/history?region=${customData.region}`),
        fetch(`${API_BASE}/api/analyze-custom`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...customData, persona })
        })
      ]);
      const analData = await analRes.json();
      const histData = await histRes.json();
      setResults(analData);
      setHistory(histData.history);
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };
"""
content = re.sub(
    r'(const handleAnalyze = async \(\) => \{)',
    handle_custom + r'\n  \1',
    content
)

# 3. Add Custom Mode UI toggle and form
custom_ui = """
          <div className="flex items-center gap-4 mb-6">
            <button 
              className={`px-4 py-2 text-sm font-semibold rounded-lg transition-colors ${!customMode ? 'bg-[#1e293b] text-white border border-slate-700 shadow-md' : 'bg-transparent text-slate-400 hover:text-slate-300'}`}
              onClick={() => { setCustomMode(false); setResults(null); }}
            >
              Historical Anomalies
            </button>
            <button 
              className={`px-4 py-2 text-sm font-semibold rounded-lg transition-colors ${customMode ? 'bg-[#1e293b] text-white border border-slate-700 shadow-md' : 'bg-transparent text-slate-400 hover:text-slate-300'}`}
              onClick={() => { setCustomMode(true); setResults(null); }}
            >
              Test Custom Data
            </button>
          </div>

          {!customMode ? (
            <div className="bg-[#1e293b] border border-slate-700 rounded-2xl p-6 shadow-xl mb-8 flex flex-col md:flex-row gap-4 items-end">
              <div className="flex-1 w-full">
                <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">Select Anomaly Event</label>
                <div className="relative">
                  <select 
                    className="w-full bg-[#0f172a] border border-slate-700 text-white rounded-xl py-3 pl-4 pr-10 appearance-none focus:outline-none focus:ring-2 focus:ring-violet-500 font-medium"
                    value={selectedScenario}
                    onChange={(e) => setSelectedScenario(e.target.value)}
                  >
                    {scenarios.map((s) => (
                      <option key={s.id} value={s.id}>{s.label}</option>
                    ))}
                  </select>
                </div>
              </div>
              
              <div className="w-full md:w-64">
                <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">Audience Persona</label>
                <div className="relative">
                  <select 
                    className="w-full bg-[#0f172a] border border-slate-700 text-white rounded-xl py-3 pl-4 pr-10 appearance-none focus:outline-none focus:ring-2 focus:ring-violet-500 font-medium"
                    value={persona}
                    onChange={(e) => setPersona(e.target.value)}
                  >
                    <option>VP of Sales</option>
                    <option>CFO</option>
                    <option>Data Scientist</option>
                    <option>Marketing Manager</option>
                  </select>
                </div>
              </div>
              
              <button 
                onClick={handleAnalyze}
                disabled={loading}
                className="w-full md:w-auto bg-violet-600 hover:bg-violet-700 disabled:bg-slate-700 disabled:text-slate-400 text-white font-bold py-3 px-8 rounded-xl transition-colors shadow-[0_0_15px_rgba(139,92,246,0.5)] flex items-center justify-center gap-2"
              >
                {loading ? "Analyzing..." : "Analyze Scenario"}
              </button>
            </div>
          ) : (
            <div className="bg-[#1e293b] border border-slate-700 rounded-2xl p-6 shadow-xl mb-8">
              <h3 className="text-sm font-bold text-slate-200 uppercase tracking-widest mb-4">Input Custom Data</h3>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
                <div>
                  <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">Region</label>
                  <select 
                    className="w-full bg-[#0f172a] border border-slate-700 text-white rounded-xl py-2 px-3 focus:outline-none focus:ring-2 focus:ring-violet-500"
                    value={customData.region}
                    onChange={(e) => setCustomData({...customData, region: e.target.value})}
                  >
                    <option>West</option><option>East</option><option>South</option><option>North</option><option>Central</option>
                    <option>Southeast</option><option>Southwest</option><option>Northeast</option><option>Midwest</option>
                  </select>
                </div>
                <div>
                  <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">Ad Spend ($)</label>
                  <input type="number" className="w-full bg-[#0f172a] border border-slate-700 text-white rounded-xl py-2 px-3 focus:outline-none focus:ring-2 focus:ring-violet-500" value={customData.ad_spend} onChange={(e) => setCustomData({...customData, ad_spend: parseFloat(e.target.value) || 0})} />
                </div>
                <div>
                  <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">Web Traffic</label>
                  <input type="number" className="w-full bg-[#0f172a] border border-slate-700 text-white rounded-xl py-2 px-3 focus:outline-none focus:ring-2 focus:ring-violet-500" value={customData.web_traffic} onChange={(e) => setCustomData({...customData, web_traffic: parseFloat(e.target.value) || 0})} />
                </div>
                <div>
                  <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">Stock On Hand</label>
                  <input type="number" className="w-full bg-[#0f172a] border border-slate-700 text-white rounded-xl py-2 px-3 focus:outline-none focus:ring-2 focus:ring-violet-500" value={customData.stock_on_hand} onChange={(e) => setCustomData({...customData, stock_on_hand: parseFloat(e.target.value) || 0})} />
                </div>
                <div>
                  <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-2">Net Revenue ($)</label>
                  <input type="number" className="w-full bg-[#0f172a] border border-slate-700 text-white rounded-xl py-2 px-3 focus:outline-none focus:ring-2 focus:ring-violet-500" value={customData.net_revenue} onChange={(e) => setCustomData({...customData, net_revenue: parseFloat(e.target.value) || 0})} />
                </div>
              </div>
              <div className="flex gap-4 justify-end">
                <select 
                  className="bg-[#0f172a] border border-slate-700 text-white rounded-xl py-2 px-4 focus:outline-none focus:ring-2 focus:ring-violet-500 font-medium"
                  value={persona}
                  onChange={(e) => setPersona(e.target.value)}
                >
                  <option>VP of Sales</option><option>CFO</option><option>Data Scientist</option><option>Marketing Manager</option>
                </select>
                <button 
                  onClick={handleAnalyzeCustom}
                  disabled={loading}
                  className="bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:text-slate-400 text-white font-bold py-2 px-8 rounded-xl transition-colors shadow-[0_0_15px_rgba(16,185,129,0.5)] flex items-center justify-center gap-2"
                >
                  {loading ? "Processing..." : "Run Pipeline"}
                </button>
              </div>
            </div>
          )}
"""
content = re.sub(
    r'<div className="bg-\[#1e293b\] border border-slate-700 rounded-2xl p-6 shadow-xl mb-8 flex flex-col md:flex-row gap-4 items-end">.*?</button>\s*</div>',
    custom_ui.replace('\\', '\\\\'),
    content,
    flags=re.DOTALL
)

# 4. Multi-Slider Simulator
multi_slider_ui = """
                  {/* WHAT-IF SIMULATOR */}
                  <div className="mt-8 bg-[#0b0f19] rounded-3xl p-8 border border-[#1e293b] shadow-inner relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-fuchsia-500 to-purple-600" />
                    <div className="flex items-center gap-4 mb-6">
                      <h2 className="text-[11px] font-black tracking-[0.2em] text-slate-400 uppercase">Interactive Causal Simulator (Multi-Node)</h2>
                    </div>
                    
                    <div className="flex flex-col gap-6">
                      {['ad_spend', 'web_traffic', 'stock_on_hand'].map(driver => {
                        const isPrimary = results.attribution.primary_driver === driver;
                        return (
                          <div key={driver}>
                            <div className="flex justify-between text-sm mb-3">
                              <span className="text-slate-400 capitalize">{driver.replace(/_/g, ' ')} {isPrimary && <span className="text-violet-400 text-xs ml-2 font-semibold">(Primary Driver)</span>}</span>
                              <span id={`sim-slider-val-${driver}`} className="text-fuchsia-400 font-bold bg-fuchsia-500/10 px-3 py-1 rounded font-mono">
                                {driver === 'web_traffic' || driver === 'stock_on_hand' ? 
                                  results.event[driver] || 0 : 
                                  `$${results.prescriptive.baseline_spend?.toLocaleString(undefined, {maximumFractionDigits:0}) || '1,500'}`
                                }
                              </span>
                            </div>
                            <input 
                              type="range" 
                              min="0" 
                              max={driver === 'stock_on_hand' ? "1000" : "5000"} 
                              step={driver === 'stock_on_hand' ? "10" : "50"}
                              defaultValue={driver === 'ad_spend' ? (results.prescriptive.baseline_spend || 1500) : (results.event[driver] || 0)}
                              id={`sim-slider-${driver}`}
                              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-fuchsia-500"
                              onChange={(e) => {
                                const newVal = parseInt(e.target.value);
                                const sliderEl = document.getElementById(`sim-slider-val-${driver}`);
                                if (sliderEl) sliderEl.innerText = (driver === 'ad_spend' ? '$' : '') + newVal.toLocaleString();
                                
                                // Debounce
                                if ((window as any).__simTimer) clearTimeout((window as any).__simTimer);
                                (window as any).__simTimer = setTimeout(async () => {
                                  const el = document.getElementById('sim-lift');
                                  if (el) el.innerText = '...';
                                  
                                  const interventions = {
                                    ad_spend: parseInt((document.getElementById('sim-slider-ad_spend') as HTMLInputElement).value),
                                    web_traffic: parseInt((document.getElementById('sim-slider-web_traffic') as HTMLInputElement).value),
                                    stock_on_hand: parseInt((document.getElementById('sim-slider-stock_on_hand') as HTMLInputElement).value)
                                  };
                                  
                                  try {
                                    const res = await fetch(`${API_BASE}/api/simulate`, {
                                      method: 'POST',
                                      headers: { 'Content-Type': 'application/json' },
                                      body: JSON.stringify({
                                        date: results.event.date,
                                        region: results.event.region,
                                        interventions: interventions
                                      })
                                    });
                                    if (res.ok) {
                                      const data = await res.json();
                                      const clampedSimulated = Math.max(0, data.simulated_revenue);
                                      const val = clampedSimulated - data.factual_revenue;
                                      const prefix = val >= 0 ? '+$' : '-$';
                                      if (el) el.innerText = prefix + Math.abs(val).toLocaleString(undefined, {maximumFractionDigits:0});
                                      
                                      const facEl = document.getElementById('sim-factual');
                                      if (facEl) facEl.innerText = '$' + data.factual_revenue.toLocaleString(undefined, {maximumFractionDigits:0});
                                      
                                      const cfEl = document.getElementById('sim-cf');
                                      if (cfEl) cfEl.innerText = '$' + clampedSimulated.toLocaleString(undefined, {maximumFractionDigits:0});
                                    }
                                  } catch (e) {
                                    // error
                                  }
                                }, 300);
                              }}
                            />
                          </div>
                        );
                      })}
                      
                      <div className="grid grid-cols-3 gap-3 mt-4 border-t border-slate-700/50 pt-6">
                        <div className="bg-[#0f172a] rounded-xl p-4 border border-slate-800 text-center">
                          <div className="text-[10px] text-slate-500 uppercase tracking-widest font-semibold mb-1">Factual Revenue</div>
                          <div id="sim-factual" className="text-lg font-bold text-slate-300">${results.event.net_revenue?.toLocaleString(undefined, {maximumFractionDigits:0})}</div>
                        </div>
                        <div className="bg-[#0f172a] rounded-xl p-4 border border-slate-800 text-center">
                          <div className="text-[10px] text-slate-500 uppercase tracking-widest font-semibold mb-1">Simulated Revenue</div>
                          <div id="sim-cf" className="text-lg font-bold text-blue-400">${results.event.net_revenue?.toLocaleString(undefined, {maximumFractionDigits:0})}</div>
                        </div>
                        <div className="bg-[#0f172a] rounded-xl p-4 border border-fuchsia-500/30 text-center">
                          <div className="text-[10px] text-slate-500 uppercase tracking-widest font-semibold mb-1">Projected Lift</div>
                          <div id="sim-lift" className="text-lg font-black text-transparent bg-clip-text bg-gradient-to-r from-fuchsia-400 to-purple-400">$0</div>
                        </div>
                      </div>
                    </div>
                  </div>
"""

# We replace the whole WHAT-IF SIMULATOR div tree.
content = re.sub(
    r'\{\/\* WHAT-IF SIMULATOR \*\/\}.*?</div>\s*</div>\s*</div>',
    multi_slider_ui.replace('\\', '\\\\'),
    content,
    flags=re.DOTALL
)

with open('frontend/src/app/page.tsx', 'w') as f:
    f.write(content)
print("Done")
