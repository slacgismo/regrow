from gld_pypower import Model

model = Model("wecc240.json")

print("\n*** Original model ***")
opf_options = {
    "curtailment_price": 10000,
    "verbose": False,
}
print("OPF options:")
for key,value in opf_options.items():
    print(f"  {key}{'.'*max([5-len(key)+len(x) for x in opf_options.keys()])} {value}")
curtailment = model.optimal_powerflow(**opf_options)["curtailment"].sum()
print(f"Curtailment before optimal sizing = {curtailment/1000:.1f} GW")

print("\n*** Optimal sizing ***")
osp_options = {
    "margin": 0.20,
    "gen_cost": 100,
    "cap_cost": 50,
    "con_cost": 500,
    "verbose": False,
}
print("OSP options:")
for key,value in osp_options.items():
    print(f"  {key}{'.'*max([5-len(key)+len(x) for x in osp_options.keys()])} {value}")
additions = model.optimal_sizing(update_model=True,**osp_options)["additions"]
model.save("wecc240_osp.json",indent=4)
# model.savecase("wecc240_osp.py")

print(f"New generation = {sum([abs(x) for x in additions['generation'].values()])/1000:.1f} GW")
print(f"New capacitors = {sum([abs(x) for x in additions['capacitors'].values() if x > 0])/1000:.1f} GW")
print(f"New condensers = {sum([abs(x) for x in additions['capacitors'].values() if x < 0])/1000:.1f} GW")

print("\n*** Optimal model ***")
curtailment = model.optimal_powerflow(**opf_options)["curtailment"].sum()
print(f"Curtailment after optimal sizing = {curtailment/1000:.1f} GW")
model.save("wecc240_opf.json",indent=4)
# model.savecase("wecc240_opf.py")

