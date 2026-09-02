with open("src/causal/dowhy_gcm.py", "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("from sklearn.ensemble import GradientBoostingRegressor", "from sklearn.ensemble import GradientBoostingRegressor\nfrom sklearn.linear_model import Ridge")

content = content.replace("""        self._model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )""", """        # Swapped from GBR to Ridge to allow linear extrapolation for counterfactual simulator out-of-distribution
        self._model = Ridge(alpha=1.0)""")

with open("src/causal/dowhy_gcm.py", "w", encoding="utf-8") as f:
    f.write(content)

print("ML Patched")
