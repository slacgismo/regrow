from arras import gld_pypower
model = gld_pypower.Model("wecc240.json")
before = model.optimal_powerflow()["curtailment"]
print(f"Curtailment before optimal sizing: {before.sum():.1f} MW")
model.optimal_sizing(margin=0.2,update_model=True)
print(f"Curtailment after optimal sizing: {after.sum():.1f} MW")
