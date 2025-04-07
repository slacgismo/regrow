import gld_pypower

print("WECC 240 Model")

model = gld_pypower.Model("wecc240.json")

gen0 = abs(model.generation().sum())
cap0 = model.capacitors().sum()
cur0 = model.optimal_powerflow()["curtailment"].sum()

model.optimal_sizing(margin=0.2,update_model=True)

gen1 = abs(model.generation().sum())
cap1 = model.capacitors().sum()
cur1 = model.optimal_powerflow()["curtailment"].sum()

print("\nBefore sizing...")
print(f"Generators: {abs(gen0):.1f} MVA")
print(f"Capacitors: {cap0:.1f} MVAr")
print(f"Curtailment: {cur0:.1f} MW")

print("\nSizing impact...")
print(f"Generators: {gen1-gen0:.1f} MVA")
print(f"Capacitors: {cap1-cap0:.1f} MVAr")
print(f"Curtailment: {cur1-cur0:.1f} MW")

print("\nAfter sizing...")
print(f"Generators: {abs(gen1):.1f} MVA")
print(f"Capacitors: {cap1:.1f} MVAr")
print(f"Curtailment: {cur1:.1f} MW")
