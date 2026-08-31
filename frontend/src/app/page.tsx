"use client";
import { useState, useEffect } from "react";
import { Activity, AlertTriangle, BrainCircuit, BarChart3, ChevronDown, CheckCircle2, LayoutDashboard, Settings, Bell, Search, Menu, HelpCircle } from "lucide-react";
import { PieChart, Pie, Cell, Tooltip as PieTooltip, ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as LineTooltip, Legend, Line } from "recharts";

const COLORS = ["#8b5cf6", "#10b981", "#3b82f6"]; // Modern SaaS colors (Violet, Emerald, Blue)

export default function Dashboard() {
  const [scenarios, setScenarios] = useState([]);
  const [selectedScenario, setSelectedScenario] = useState("");
  const [persona, setPersona] = useState("VP of Sales");
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [history, setHistory] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("http://localhost:8000/api/scenarios")
      .then((res) => res.json())
      .then((data) => {
        setScenarios(data.scenarios);
        if (data.scenarios.length > 0) {
          setSelectedScenario(data.scenarios[0].id);
        }
      })
      .catch((err) => console.error("Failed to load scenarios:", err));
  }, []);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setResults(null);
    setHistory(null);
    try {
      const scenario = scenarios.find((s) => s.id === selectedScenario);
      
      const [histRes, analRes] = await Promise.all([
        fetch(`http://localhost:8000/api/history?region=${scenario.region}`),
        fetch("http://localhost:8000/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            date: scenario.date,
            region: scenario.region,
            persona: persona,
          }),
        })
      ]);

      if (!analRes.ok) throw new Error("API failed to respond.");
      
      const analData = await analRes.json();
      const histData = await histRes.json();
      
      setResults(analData);
      setHistory(histData.history);
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  const renderAttributionChart = () => {
    if (!results) return null;
    const { attribution } = results;
    const data = [
      { name: "Ad Spend", value: attribution.ad_spend_contribution_pct || 0 },
      { name: "Inventory", value: attribution.stock_on_hand_contribution_pct || 0 },
      { name: "Web Traffic", value: attribution.web_traffic_contribution_pct || 0 },
    ].filter((d) => d.value > 0);

    return (
      <div className="h-52 w-full mt-2">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={data} innerRadius={60} outerRadius={80} paddingAngle={8} dataKey="value" stroke="none" cornerRadius={4}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <PieTooltip 
              formatter={(value) => `${value.toFixed(1)}%`}
              contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', color: '#f8fafc', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)' }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderTimeSeries = () => {
    if (!history) return null;
    return (
      <div className="bg-[#0f172a] border border-[#1e293b] rounded-2xl p-6 shadow-2xl mb-8 relative overflow-hidden group">
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-violet-600 via-blue-500 to-emerald-400 opacity-50 group-hover:opacity-100 transition-opacity" />
        <div className="flex justify-between items-center mb-6">
          <div>
            <h3 className="text-sm font-bold text-slate-200 uppercase tracking-widest">Revenue Baseline vs Actual</h3>
            <p className="text-xs text-slate-500 mt-1">STL Decomposition Anomaly Tracking</p>
          </div>
          <div className="bg-[#1e293b] px-3 py-1 rounded-full text-xs text-slate-300 font-medium">12 Month History</div>
        </div>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={history} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              <XAxis dataKey="date" stroke="#64748b" fontSize={11} tickMargin={12} minTickGap={40} axisLine={false} tickLine={false} />
              <YAxis stroke="#64748b" fontSize={11} tickFormatter={(val) => `$${val/1000}k`} axisLine={false} tickLine={false} />
              <LineTooltip 
                contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '12px', color: '#f8fafc', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)' }}
                labelStyle={{ color: '#94a3b8', marginBottom: '8px', fontWeight: 600 }}
                formatter={(value) => `$${Number(value).toLocaleString()}`}
              />
              <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '20px' }} iconType="circle" />
              <Area type="monotone" dataKey="actual" name="Actual Revenue" stroke="#8b5cf6" strokeWidth={3} fillOpacity={1} fill="url(#colorActual)" activeDot={{ r: 6, strokeWidth: 0, fill: '#fff' }} />
              <Line type="monotone" dataKey="baseline" name="Expected Baseline" stroke="#10b981" strokeWidth={2} strokeDasharray="6 6" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-[#020617] text-slate-200 font-sans flex flex-col selection:bg-violet-500/30">
      
      {/* Top Navigation */}
      <header className="h-16 border-b border-[#1e293b] bg-[#0b0f19]/80 backdrop-blur-md flex items-center justify-between px-6 sticky top-0 z-50">
        <div className="flex items-center gap-8">
          <div className="flex items-center gap-3">
            <div className="bg-violet-600 p-2 rounded-lg shadow-[0_0_15px_rgba(124,58,237,0.5)]">
              <Activity className="text-white" size={20} />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-white">KPI Engine</h1>
          </div>
        </div>
        <div className="flex items-center gap-4 text-slate-400">
          <div className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-3 py-1 rounded-full text-xs font-semibold tracking-wide flex items-center gap-2">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            LIVE DEMO
          </div>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-80 border-r border-[#1e293b] bg-[#0b0f19] p-6 flex flex-col overflow-y-auto">
          <div className="mb-8">
            <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4">Command Center</h2>
            
            <div className="space-y-5">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-2">Access Persona</label>
                <div className="relative group">
                  <select 
                    value={persona} 
                    onChange={(e) => setPersona(e.target.value)}
                    className="w-full bg-[#0f172a] border border-[#1e293b] group-hover:border-violet-500/50 rounded-xl p-3.5 text-sm appearance-none focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition-all shadow-sm"
                  >
                    <option value="VP of Sales">VP of Sales (Strategic)</option>
                    <option value="Regional Manager">Regional Manager (Tactical)</option>
                  </select>
                  <ChevronDown className="absolute right-4 top-4 text-slate-500 pointer-events-none transition-transform group-hover:text-violet-400" size={16} />
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-2">Anomaly Scenario</label>
                <div className="relative group">
                  <select 
                    value={selectedScenario} 
                    onChange={(e) => setSelectedScenario(e.target.value)}
                    className="w-full bg-[#0f172a] border border-[#1e293b] group-hover:border-violet-500/50 rounded-xl p-3.5 text-sm appearance-none focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 transition-all shadow-sm"
                  >
                    {scenarios.map((s) => (
                      <option key={s.id} value={s.id}>{s.label}</option>
                    ))}
                  </select>
                  <ChevronDown className="absolute right-4 top-4 text-slate-500 pointer-events-none transition-transform group-hover:text-violet-400" size={16} />
                </div>
              </div>

              <button 
                onClick={handleAnalyze} 
                disabled={loading || scenarios.length === 0}
                className="w-full mt-2 bg-white text-slate-900 hover:bg-slate-100 disabled:opacity-50 font-semibold py-3.5 rounded-xl flex justify-center items-center gap-2 transition-all shadow-lg active:scale-[0.98]"
              >
                {loading ? (
                  <span className="animate-pulse flex items-center gap-2"><div className="w-4 h-4 border-2 border-slate-900/30 border-t-slate-900 rounded-full animate-spin" /> Processing...</span>
                ) : (
                  <>
                    <BrainCircuit size={18} className="text-violet-600" />
                    Execute Causal Engine
                  </>
                )}
              </button>
            </div>
          </div>
          
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 p-8 overflow-y-auto bg-gradient-to-br from-[#020617] to-[#0b0f19]">
          <div className="max-w-6xl mx-auto">
            
            {/* Empty State */}
            {!results && !loading && (
              <div className="h-[70vh] flex flex-col items-center justify-center text-slate-500">
                <div className="w-24 h-24 bg-[#0f172a] rounded-full flex items-center justify-center mb-6 border border-[#1e293b] shadow-2xl">
                  <BarChart3 size={40} className="text-slate-600" />
                </div>
                <h2 className="text-2xl font-semibold text-slate-300">Ready for Analysis</h2>
                <p className="text-sm mt-3 max-w-md text-center leading-relaxed">Select a detected anomaly from the command center to initiate counterfactual decomposition and CATE estimation.</p>
              </div>
            )}

            {/* Skeleton Loading State */}
            {loading && (
              <div className="space-y-8 animate-pulse">
                <div className="h-10 bg-[#1e293b] rounded-lg w-1/3" />
                <div className="h-72 bg-[#0f172a] rounded-2xl border border-[#1e293b]" />
                <div className="grid grid-cols-3 gap-8">
                  <div className="col-span-1 h-64 bg-[#0f172a] rounded-2xl border border-[#1e293b]" />
                  <div className="col-span-2 h-64 bg-[#0f172a] rounded-2xl border border-[#1e293b]" />
                </div>
              </div>
            )}

            {/* Results State */}
            {results && !loading && (
              <div className="animate-in fade-in slide-in-from-bottom-8 duration-700 space-y-8">
                
                {/* Dashboard Header */}
                <div className="flex justify-between items-end">
                  <div>
                    <div className="flex items-center gap-3 mb-2">
                      <span className="bg-slate-800 text-slate-300 text-xs px-2.5 py-1 rounded-md font-medium tracking-wide">REGION: {results.event.region.toUpperCase()}</span>
                      <span className="bg-red-500/10 text-red-400 border border-red-500/20 text-xs px-2.5 py-1 rounded-md font-medium tracking-wide">STATUS: {results.event.severity.toUpperCase()}</span>
                    </div>
                    <h2 className="text-3xl font-bold tracking-tight text-white">{results.event.date} Anomaly Profile</h2>
                  </div>
                  <div className="text-right bg-[#0f172a] border border-[#1e293b] px-6 py-3 rounded-2xl shadow-lg">
                    <div className="text-xs text-slate-500 uppercase tracking-widest font-semibold mb-1">KPI Deviation</div>
                    <div className="text-2xl font-black text-red-500">
                      {results.event.pct_deviation ? (results.event.pct_deviation * 100).toFixed(1) : 0}%
                    </div>
                  </div>
                </div>

                {renderTimeSeries()}

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                  
                  {/* Causal Attribution */}
                  <div className="col-span-1 bg-[#0f172a] border border-[#1e293b] rounded-2xl p-6 shadow-xl flex flex-col relative overflow-hidden group">
                    <div className="absolute inset-0 bg-gradient-to-br from-violet-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                    <div>
                      <h3 className="text-sm font-bold text-slate-200 uppercase tracking-widest mb-1">Root Cause Tracing</h3>
                      <p className="text-xs text-slate-500">Shapley Counterfactuals</p>
                    </div>
                    
                    <div className="flex-1 flex flex-col justify-center">
                      {renderAttributionChart()}
                    </div>

                    <div className="mt-4 pt-4 border-t border-[#1e293b] text-center">
                      <div className="text-[10px] text-slate-500 uppercase tracking-widest font-semibold mb-1">Primary Driver Identified</div>
                      <div className="text-lg font-bold text-violet-400 capitalize bg-violet-500/10 py-1.5 rounded-lg border border-violet-500/20">
                        {results.attribution.primary_driver?.replace(/_/g, ' ')}
                      </div>
                    </div>
                  </div>

                  {/* Prescriptive Lift */}
                  <div className="col-span-2 bg-[#0f172a] border border-[#1e293b] rounded-2xl p-6 shadow-xl relative overflow-hidden flex flex-col">
                    <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-5 mix-blend-overlay" />
                    <div>
                      <h3 className="text-sm font-bold text-slate-200 uppercase tracking-widest mb-1">Prescriptive Intelligence</h3>
                      <p className="text-xs text-slate-500">Double Machine Learning (CATE)</p>
                    </div>
                    
                    {results.prescriptive.data_ambiguity ? (
                       <div className="flex-1 flex flex-col items-center justify-center text-center mt-6 bg-red-950/20 rounded-xl border border-red-900/50 p-8">
                         <AlertTriangle className="text-red-500 mb-3" size={40} />
                         <h4 className="font-bold text-lg text-red-400 tracking-wide">Data Ambiguity Triggered</h4>
                         <p className="text-sm text-red-300/80 max-w-sm mt-2 leading-relaxed">System abstaining from prescriptive logic. History is too sparse (&lt;30 days) to calculate causal interventions safely.</p>
                       </div>
                    ) : (
                      <div className="flex-1 flex flex-col justify-center items-center mt-4">
                        <div className="text-xs text-emerald-500 uppercase tracking-widest font-bold mb-2">Expected Revenue Recovery</div>
                        <div className="text-6xl font-black text-transparent bg-clip-text bg-gradient-to-b from-emerald-400 to-emerald-600 drop-shadow-sm">
                          ${(results.prescriptive.expected_lift || 0).toLocaleString(undefined, {maximumFractionDigits: 0})}
                        </div>
                        <div className="text-sm text-slate-400 mt-3 font-medium bg-[#1e293b] px-4 py-1.5 rounded-full border border-slate-700">
                          If {results.attribution.primary_driver?.replace(/_/g, ' ')} is restored
                        </div>
                        
                        <div className="w-full flex justify-center gap-12 mt-10">
                          <div className="text-center bg-[#0b0f19] px-8 py-4 rounded-2xl border border-[#1e293b] shadow-inner">
                            <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-1">Current State</div>
                            <div className="text-xl font-bold text-red-400">${(results.prescriptive.current_spend || 0).toLocaleString(undefined, {maximumFractionDigits:0})}</div>
                          </div>
                          <div className="text-center bg-[#0b0f19] px-8 py-4 rounded-2xl border border-[#1e293b] shadow-inner">
                            <div className="text-[10px] text-slate-500 uppercase tracking-widest font-bold mb-1">Target Baseline</div>
                            <div className="text-xl font-bold text-emerald-400">${(results.prescriptive.baseline_spend || 0).toLocaleString(undefined, {maximumFractionDigits:0})}</div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                
                  {/* WHAT-IF SIMULATOR */}
                  <div className="mt-8 bg-[#0b0f19] rounded-3xl p-8 border border-[#1e293b] shadow-inner relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-fuchsia-500 to-purple-600" />
                    <div className="flex items-center gap-4 mb-6">
                      <h2 className="text-[11px] font-black tracking-[0.2em] text-slate-400 uppercase">Interactive What-If Simulator</h2>
                      <div className="bg-fuchsia-500/10 text-fuchsia-400 border border-fuchsia-500/20 px-2 py-0.5 rounded text-[10px] font-bold">BETA</div>
                    </div>
                    
                    <div className="flex flex-col gap-6">
                      <div>
                        <div className="flex justify-between text-sm mb-3">
                          <span className="text-slate-400">Intervene on <span className="font-bold text-white capitalize">{results.attribution.primary_driver?.replace(/_/g, ' ')}</span></span>
                          <span className="text-fuchsia-400 font-bold bg-fuchsia-500/10 px-2 py-1 rounded">Live API</span>
                        </div>
                        <input 
                          type="range" 
                          min="0" 
                          max="5000" 
                          defaultValue={results.prescriptive.baseline_spend || 1500}
                          className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-fuchsia-500"
                          onChange={async (e) => {
                            const newVal = parseInt(e.target.value);
                            const el = document.getElementById('sim-lift');
                            if (el) el.innerText = 'Calculating...';
                            
                            try {
                              const res = await fetch('http://localhost:8000/api/simulate', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({
                                  date: results.event.date,
                                  region: results.event.region,
                                  driver: results.attribution.primary_driver,
                                  new_value: newVal
                                })
                              });
                              if (res.ok) {
                                const data = await res.json();
                                if (el) el.innerText = '+$' + Math.max(0, data.lift).toLocaleString(undefined, {maximumFractionDigits:0});
                              }
                            } catch (e) {
                              // error
                            }
                          }}
                        />
                        <div className="flex justify-between text-[10px] text-slate-500 mt-2 font-mono">
                          <span>$0</span>
                          <span>$5,000</span>
                        </div>
                      </div>
                      
                      <div className="bg-[#0f172a] rounded-xl p-4 border border-slate-800 flex items-center justify-between">
                        <div className="text-xs text-slate-400">Projected Revenue Lift</div>
                        <div id="sim-lift" className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-fuchsia-400 to-purple-400">
                          +${(results.prescriptive.expected_lift || 0).toLocaleString(undefined, {maximumFractionDigits:0})}
                        </div>
                      </div>
                    </div>
                  </div>


                {/* AI Synthesis */}
                <div className="bg-[#1e293b] border border-slate-700 rounded-2xl p-8 shadow-2xl relative overflow-hidden">
                   <div className="absolute top-0 right-0 w-64 h-64 bg-blue-500/10 blur-[100px] rounded-full pointer-events-none" />
                   
                   <div className="flex items-center justify-between mb-6 pb-6 border-b border-slate-700">
                      <div className="flex items-center gap-3">
                        <div className="bg-blue-500/20 p-2 rounded-lg border border-blue-500/30">
                          <BrainCircuit className="text-blue-400" size={20} />
                        </div>
                        <h3 className="text-base font-bold text-slate-200 uppercase tracking-widest">Generative Synthesis</h3>
                      </div>
                      <span className="text-xs bg-[#0f172a] text-blue-400 px-3 py-1.5 rounded-full border border-blue-500/30 font-medium">
                        Model: {results.telemetry.model_used}
                      </span>
                   </div>
                   
                   <p className="text-slate-200 text-lg leading-relaxed font-light mb-8 max-w-4xl">
                     {results.narrative.narrative_summary}
                   </p>

                   <div className="grid grid-cols-1 md:grid-cols-2 gap-10">
                     <div className="bg-[#0f172a] rounded-xl p-6 border border-slate-800 shadow-inner">
                       <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                         <BarChart3 size={14} /> Mathematical Drivers
                       </h4>
                       <ul className="space-y-3">
                         {results.narrative.key_drivers.map((driver, idx) => (
                           <li key={idx} className="flex items-start gap-3 text-sm text-slate-300">
                             <div className="mt-1.5 w-1.5 h-1.5 rounded-full bg-violet-500 shrink-0 shadow-[0_0_8px_rgba(139,92,246,0.8)]" />
                             <span className="leading-relaxed">{driver}</span>
                           </li>
                         ))}
                       </ul>
                     </div>
                     <div className="bg-[#0f172a] rounded-xl p-6 border border-slate-800 shadow-inner">
                       <h4 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                         <Activity size={14} /> Recommended Action Plan
                       </h4>
                       <ul className="space-y-3">
                         {results.narrative.recommended_actions.map((action, idx) => (
                           <li key={idx} className="flex items-start gap-3 text-sm text-slate-300">
                             <CheckCircle2 className="mt-0.5 text-emerald-500 shrink-0" size={16} />
                             <span className="leading-relaxed">{action}</span>
                           </li>
                         ))}
                       </ul>
                     </div>
                   </div>
                </div>

                {/* Audit Footer */}
                <div className="flex justify-between items-center text-[10px] text-slate-500 uppercase tracking-widest font-semibold px-2 pb-8">
                   <div className="flex items-center gap-6">
                     <span className="flex items-center gap-2">
                       <span className="relative flex h-2 w-2">
                         <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                         <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                       </span>
                       System Nominal
                     </span>
                     <span>End-to-End Latency: <span className="text-slate-400">{results.telemetry.latency_seconds?.toFixed(2)}s</span></span>
                   </div>
                   <div>
                     Tokens: <span className="text-slate-400">{results.telemetry.prompt_tokens} In</span> / <span className="text-slate-400">{results.telemetry.completion_tokens} Out</span>
                   </div>
                </div>

              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
