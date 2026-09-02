with open("src/api/main.py", "r") as f:
    content = f.read()

endpoint_code = """
class SimulateRequest(BaseModel):
    date: str
    region: str
    driver: str
    new_value: float

@app.post("/api/simulate")
async def simulate_scenario(req: SimulateRequest):
    from src.causal.simulator import run_simulation
    try:
        res = run_simulation(PROJECT_ROOT, req.date, req.region, req.driver, req.new_value)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""

if "class SimulateRequest" not in content:
    content += "\n" + endpoint_code + "\n"
    with open("src/api/main.py", "w") as f:
        f.write(content)
    print("API Patched successfully")
else:
    print("API already patched")
