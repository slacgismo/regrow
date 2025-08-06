import __init__

if __name__ == "__main__":

    if not os.path.exists("example.json"):
        print("TEST: example.json not found, testing not done",file=sys.stderr)
        quit()

    test = Model("example.json")

    try:
        test.run("--version")
        runtime=True
    except RuntimeError as err:
        print("\nWARNING: GridLAB-D not installed, skipping runtime tests",file=sys.stderr)
        runtime = False

    tested = 0
    failed = 0

    def testEq(a,b,msg):
        import inspect
        global tested
        tested += 1
        if a != b:
            caller = inspect.getframeinfo(inspect.stack()[1][0])
            print(f"TEST [{os.path.basename(caller.filename)}@{caller.lineno}]: {msg}: {repr(a)} != {repr(b)}",file=sys.stderr,flush=True)
            global failed
            failed += 1
    def testException(a,exc,msg):
        try:
            a()
            testEq(None,exc.__name__,msg)
        except:
            e_type,e_value,e_trace = sys.exc_info()
            testEq(e_type.__name__,exc.__name__,msg)


    if runtime:
        testEq(test.run(),"","initial run test failed")

    bus_3 = test.get_object("bus_3")

    # accessor tests
    print("TEST: testing accessors",file=sys.stderr,flush=True)
    testEq(test.property("bus_0",'id'),2,"get header failed")
    testEq(bus_3["bus_i"],'4',"get object failed")
    testException(lambda:test.add_object("bus","bus_3",**bus_3)["bus_i"],ValueError,"add object succeeded")
    testException(lambda:test.add_object("bus","bus_4",id="0")["bus_i"],ValueError,"add object succeeded")
    testEq(test.del_object("bus_3"),bus_3,"del object failed")
    testEq(test.add_object("bus","bus_3",**bus_3)["bus_i"],bus_3["bus_i"],"add object failed")
    testException(lambda:test.add_object("bus","bus_0"),ValueError,"add object failed")
    testException(lambda:test.add_object("transformer","test"),ValueError,"add object failed")
    testEq(test.add_object("geodata","test",scale=0.1),{'class': 'geodata', 'id': "13", 'scale': '0.1 pu'},"add object failed")
    testEq(test.mod_object("test",scale=1.0),{'class': 'geodata', 'id': "13", 'scale': '1.0 pu'},"mod object failed")
    testEq(test.del_object("test"),{'class': 'geodata', 'id': "13", 'scale': '1.0 pu'},"add object failed")

    # content tests
    print("TEST: testing model contents",file=sys.stderr,flush=True)
    testEq("pypower" in test.modules(),True,"module failed")
    testEq(test.validate(["pypower"]),None, "validate failed")
    testEq("version" in test.globals(list),True,"globals list failed")
    testEq(test.globals(dict)["country"],"US", "globals dict failed" )
    testEq(test.globals("country"),"US", "globals get failed")
    testEq(test.find("bus",list),['bus_0', 'bus_1', 'bus_2', 'bus_3'], "find list failed")
    testEq([y['bus_i'] for y in test.find("bus",dict).values()],['1','2','3','4'], "find dict failed")
    testEq(list(test.select({"class":"bus","type":"REF"})),['bus_0'],"select failed")
    testEq(test.get_name('bus') , ['bus_0', 'bus_1', 'bus_2', 'bus_3'], "get bus name failed")
    testEq(test.get_name('bus',0) , 'bus_0', "get bus name failed")
    testEq(test.get_name('bus',[1,2]) , ['bus_1', 'bus_2'], "get bus name failed")
    testEq(test.get_name('branch') , ['branch:6', 'branch:7', 'branch:8'], "get branch failed")
    testEq(test.get_name('branch',0) , 'branch:6', "get branch failed")
    testEq(test.get_name('branch',[1,2]) , ['branch:7', 'branch:8'], "get branch failed")
    testEq(test.get_bus("gen_0") , "bus_0", "get bus failed")
    testEq(test.get_bus(["gen_0"]) , ["bus_0"], "get bus failed")
    testEq(test.property("bus_0","Pd"),0.0, "property float failed")
    testEq(test.property("bus_0","S"),0j, "property complex failed")
    testEq(test.perunit("S"),100, "perunit power failed")
    testEq(test.perunit("V"),[12.5, 12.5, 12.5, 12.5], "perunit voltage failed")
    testEq(test.perunit("Z"),[1.5625, 1.5625, 1.5625], "perunit impedance failed")
    testEq(test.graphLaplacian().shape,(4,4), "graph Laplacian failed")
    testEq(test.graphIncidence().shape,(3,4), "graph incidence failed")
    testEq(test.demand().tolist(),[0j,0j,0.1+0.01j,0.1+0.01j], "demand failed")
    testEq(list(test.generators().keys()) , ['gen_0'], "generators failed")
    testEq(test.generation().tolist() , [(0.1+0.05j), 0j, 0j, 0j], "generation failed")
    testEq(list(test.costs().keys()) , ['gencost:1'], "costs failed")
    testEq(test.prices().tolist() , [0,0,0,0], "prices failed")
    testEq(test.lineratings().tolist() , [0.25,0.15,0.15], "line ratings failed")
    testEq(test.capacitors().tolist() , [0,0,0,0], "capacitors failed")
    testEq(test.mermaid().split("\n")[0],"graph TB","mermaid failed")

    # optimization tests
    print("TEST: testing optimizations",file=sys.stderr,flush=True)
    testEq(test.optimal_powerflow()["curtailment"].round(1).tolist(),[0.0, 0.0, 6.8, 6.8],"optimal powerflow failed")
    testEq(test.optimal_sizing(refresh=True,gen_cost=np.array([100,500,1000,1000])+1000j,cap_cost={0:1000,1:500})["generation"].round(1).tolist() , [(26.4+0j), 0j, 0j, 0j], "optimal sizing failed")
    testEq(test.optimal_sizing(refresh=True,gen_cost=np.array([100,500,1000,1000])+1000j,cap_cost={0:1000,1:500})["capacitors"].round(1).tolist() , [0,0,1.2,1.2], "optimal sizing failed")
    testEq(test.optimal_sizing(refresh=True,gen_cost=np.array([100,500,1000,1000])+1000j,cap_cost={0:1000,1:500},update_model=True)["additions"] , {'generation': {0: (16.4+0j)}, 'capacitors': {2: 1.2, 3: 1.2}} , "optimal sizing failed")
    testEq(test.optimal_powerflow(refresh=True)["curtailment"].tolist(),[0,0,0,0],"optimal powerflow failed")
    if runtime:
        test.data["globals"]["savefile"] = ""
        test.save("test_out.json",indent=4)
        rc,out,err = test.run("test_out.json",exception=False)
        testEq(out,[''],'run test failed')

    # case tests
    print("TEST: testing pypower cases",file=sys.stderr,flush=True)
    for file in sorted(os.listdir("test")):
        if file.startswith("case") and file.endswith(".json"):

            test = Model(os.path.join("test",file))

            # pypower PF test
            try:
                testEq(test.runpf(OUT_ALL=0,VERBOSE=0)[1],1,f"{file} runpf failed")
            except Exception as err:
                print(f"ERROR: {file} runpf raised exception {err}",file=sys.stderr)
                test.savecase("test/"+file.replace(".json","_runpf_failed.py"))
                failed += 1

            # pypower OPF test
            try:
                if test.find("gencost"):
                    testEq(test.runopf(OUT_ALL=0,VERBOSE=0)["success"],True,f"{file} runopf failed")
            except Exception as err:
                print(f"ERROR: {file} runopf raised exception {err}",file=sys.stderr)
                test.savecase("test/"+file.replace(".json","_runopf_failed.py"))
                failed += 1

            # enhanced OPF test
            if not test.optimal_powerflow(on_fail=lambda x:print(f"\nTEST: {file} initial OPF is {x}",file=sys.stderr)):
                test.optimal_powerflow(verbose=True,on_fail=lambda x: print(test.problem,file=sys.stderr))
                failed += 1
            tested += 1

            # OSP test
            testEq(test.optimal_sizing(refresh=True,update_model=True)["status"],"optimal","sizing failed")

            # OSP/OPF test
            testEq(test.optimal_powerflow(refresh=True)["curtailment"].tolist(),np.zeros(len(test.find("bus"))).tolist(),"final OPF failed")

    print("TEST: completed",tested,"tests",file=sys.stderr,flush=True)
    if failed:
        print("ERROR:",failed,"test failed",file=sys.stderr)
    else:
        print("TEST: no errors",file=sys.stderr)
