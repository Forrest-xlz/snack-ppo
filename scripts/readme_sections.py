"""Model diagrams, feature dictionaries and result tables for the README."""

MLP_DIAGRAM = '''```mermaid
flowchart TD
    X["Head / body / food: N x 3 x 12 x 12"] --> F["Flatten: N x 432"]
    F --> H1["Linear 432 to 256 + ReLU"]
    H1 --> H2["Linear 256 to 256 + ReLU"]
    H2 --> P["Actor: Linear 256 to 4 + softmax"]
    H2 --> V["Critic: Linear 256 to 1"]
```'''

CNN_DIAGRAM = '''```mermaid
flowchart TD
    X["Original features: N x 3 x 12 x 12"] --> C1["Conv 3 to 16, 3x3, stride 2, padding 1 + ReLU: N x 16 x 6 x 6"]
    C1 --> C2["Conv 16 to 32, 3x3, stride 2, padding 1 + ReLU: N x 32 x 3 x 3"]
    C2 --> C3["Conv 32 to 64, 3x3, stride 1, padding 1 + ReLU: N x 64 x 3 x 3"]
    C3 --> F["Flatten: N x 576"]
    F --> H["Linear 576 to 256 + ReLU"]
    H --> P["Actor: Linear 256 to 4 + softmax"]
    H --> V["Critic: Linear 256 to 1"]
```'''

GRID_DIAGRAM = '''```mermaid
flowchart TD
    X["Grid features: N x 144 x 7"] --> E["Linear 7 to 32 + learned cell positions"]
    E --> T["Append learned critic token: N x 145 x 32"]
    T --> E1["Encoder 1: 145 queries, 2 heads, FFN 32 to 64 to 32"]
    E1 --> E2["Encoder 2: head-cell / critic queries; all 145 keys and values"]
    E2 --> N["Final LayerNorm: N x 2 x 32"]
    N --> H["Shared Linear 32 to 256 + ReLU, applied to each readout"]
    H --> P["Head-cell row: Linear 256 to 4 + softmax"]
    H --> V["Critic-token row: Linear 256 to 1"]
```'''


def entity_diagram(dim, hidden, layers):
    middle = (f'    T --> E1["Encoder 1: all 147 tokens, 2 heads, FFN {dim} to {2*dim} to {dim}"]\n'
              if layers == 2 else
              f'    T --> E1["Encoder 1 and Encoder 2: all 147 tokens, 2 heads, FFN {dim} to {2*dim} to {dim}"]\n')
    return f'''```mermaid
flowchart TD
    C["144 Cell tokens: 66 features each"] --> CM["Cell MLP: 66 to {hidden} to {dim}, ReLU"]
    S["1 Snake token: 21 features"] --> SM["Snake MLP: 21 to {hidden} to {dim}, ReLU"]
    F["1 Food token: 24 features"] --> FM["Food MLP: 24 to {hidden} to {dim}, ReLU"]
    G["1 Global token: 8 features"] --> GM["Global MLP: 8 to {hidden} to {dim}, ReLU"]
    CM --> T["147 tokens + type embeddings; learned positions on Cell tokens"]
    SM --> T
    FM --> T
    GM --> T
{middle}    E1 --> LAST["Encoder {layers}: Snake / Global queries; all 147 keys and values"]
    LAST --> LN["Final LayerNorm: N x 2 x {dim}"]
    LN --> H["Shared Linear {dim} to 256 + ReLU, applied to each readout"]
    H --> P["Snake row: Linear 256 to 4 + softmax"]
    H --> V["Global row: Linear 256 to 1"]
```'''


TEACHER_DIAGRAM = '''```mermaid
flowchart LR
    X["Student rollout states"] --> S["Trainable small Transformer"]
    X --> T["Frozen small Transformer teacher"]
    S --> PPO["PPO policy + value loss - entropy bonus"]
    S --> KL["KL teacher to student, coefficient 0.001"]
    T --> KL
    PPO --> L["Total loss: update student only"]
    KL --> L
    S --> E["Evaluation every 25M samples"]
    E --> B["Strictly higher mean score"]
    B --> COPY["Copy student weights to teacher"]
    COPY --> T
```'''


def feature_rows():
    cells = [(i, name, "0/1") for i, name in enumerate([
        "Snake head", "Ordinary body (excluding head and tail)", "Food", "Empty cell", "Snake tail"
    ])]
    cells.append((5, "Normalized release time", "t / A"))
    cells.extend((5+t, f"Release in exactly {t} steps", f"1 if t={t}, else 0") for t in range(1,17))
    cells.append((22, "Release later than 16 steps", "1 if t>16, else 0"))
    cells.extend((23+i, f"Log-release node {node}", "Log-space two-hot weight") for i,node in enumerate([1,2,4,8,16,32,64,128,144]))
    cells.extend([(32,"Absolute x","x / 11"),(33,"Absolute y","y / 11"),
                  (34,"Ego x","dx′ / 11"),(35,"Ego y","dy′ / 11")])
    def fourier(start):
        rows=[]
        for j,freq in enumerate([1,2,4,8]):
            for k,(axis,func) in enumerate([("x","sin"),("x","cos"),("y","sin"),("y","cos")]):
                offset="dx′" if axis=="x" else "dy′"
                rows.append((start+4*j+k,f"Ego {axis} {func}, frequency {freq}",f"{func}(π × {freq} × {offset} / 12)"))
        return rows
    cells.extend(fourier(36))
    for start,endpoint in [(52,"head"),(56,"tail")]:
        cells.extend((start+i,f"Link toward {endpoint}: ego {direction}","0/1; zero when link is absent") for i,direction in enumerate(["up","down","left","right"]))
    cells.extend([(60,"Region size","s / A"),(61,"Log region size","log(1+s) / log(1+A)"),
                  (62,"Region size / snake length","min(s/L, 2)"),(63,"Region size >= snake length","1 if s>=L, else 0"),
                  (64,"Region adjacent to tail","0/1"),(65,"Region adjacent to head","0/1")])
    snake=[(0,"Normalized snake length","L / A"),(1,"Log snake length","log(L) / log(A)")]
    snake.extend((2+i,f"Length node {node}","Linear two-hot weight") for i,node in enumerate([3,5,10,20,40,60,90,120,144]))
    snake.extend([(11,"Remaining free cells","(A-L) / A"),(12,"Log remaining free cells","log(1+A-L) / log(1+A)")])
    snake.extend((13+i,f"Absolute heading: {direction}","0/1") for i,direction in enumerate(["up","down","left","right"]))
    snake.extend([(17,"Head absolute x","head_x / 11"),(18,"Head absolute y","head_y / 11"),
                  (19,"Largest head-adjacent region","h / A"),(20,"Largest head-adjacent region / length","min(h/L, 2)")])
    food=[(0,"Food exists","0/1"),(1,"Food absolute x","food_x / 11"),(2,"Food absolute y","food_y / 11"),
          (3,"Food ego x","dx′ / 11"),(4,"Food ego y","dy′ / 11")]
    food.extend(fourier(5))
    food.extend([(21,"Food-region size","s_food / A"),(22,"Food-region size / length","min(s_food/L, 2)"),
                 (23,"Food region adjacent to head","0/1")])
    global_rows=[(0,"Normalized snake length","L / A"),(1,"Remaining free cells","(A-L) / A"),
                 (2,"Log remaining free cells","log(1+A-L) / log(1+A)")]
    global_rows.extend((3+i,f"Exactly {k} connected regions",f"1 if K={k}, else 0") for i,k in enumerate([1,2,3,4]))
    global_rows.append((7,"At least 5 connected regions","1 if K>=5, else 0"))
    for rows,size in [(cells,66),(snake,21),(food,24),(global_rows,8)]:
        assert [r[0] for r in rows]==list(range(size))
    return cells,snake,food,global_rows


def blocks(results, lookup, configs, at, number, millions, table):
    base={k:lookup[("E001",k)] for k in "ABCDEF"}
    early=[];late=[]
    for key in "ABC":
        for epoch in [1,2,4,8]:
            r=base[key] if epoch==1 else lookup[("E002",f"{key}-ep{epoch}")]
            early.append([key,epoch,*[number(at(r,t)) for t in [25,50,100]]])
            late.append([key,epoch,number(r.eval_at_300M),number(r.best_eval),number(r.last_eval),number(r.last4_eval_mean),millions(r.last_logged_samples)])
    switches=[]
    for r in results[results.study=="E003"].itertuples(index=False):
        switches.append([r.variant,number(r.initial_eval),number(r.eval_300M_after_switch),number(r.best_eval),number(r.last_eval),f"{millions(r.last_logged_samples)} / {millions(r.last_logged_samples-r.pretrained_samples)}"])
    features=feature_rows()
    model_config=[];architecture_results=[]
    specs={
        "A":("CNN","3 planes","—","3 convolutions","—","172,581"),
        "B":("Grid Transformer v3","7 / Cell","7 → 32","2","32","31,781"),
        "B-v7":("Entity Transformer small","66 / 21 / 24 / 8","input → 32 → 16","2","16","18,533"),
        "C":("Entity Transformer base","66 / 21 / 24 / 8","input → 64 → 32","3","32","56,357"),
        "D":("Small + teacher KL","66 / 21 / 24 / 8","input → 32 → 16","2","16","18,533 per model"),
    }
    for key in ["A","B","B-v7","C","D"]:
        r=lookup[("E004",key)]
        model_config.append([key,*specs[key],"0.001" if key=="D" else "0"])
        architecture_results.append([key,number(r.eval_at_300M),number(r.best_eval),number(r.last_eval),number(r.last4_eval_mean),f"{millions(r.last_logged_samples)} / {millions(r.last_eval_samples)}"])
    original=[(0,"Snake head","1 at the head cell, 0 elsewhere"),(1,"Snake body","1 on all body cells including the tail, excluding the head"),(2,"Food","1 at the food cell, 0 elsewhere")]
    grid=original+[(3,"Absolute x","x / 11"),(4,"Absolute y","y / 11"),(5,"Empty cell","1 if no head, body or food occupies the cell"),(6,"Distance along the body from the tail","Tail index=0; increasing toward the head; divide by 143; non-snake cells=0")]
    return {
        "MLP_DIAGRAM":MLP_DIAGRAM,"CNN_DIAGRAM":CNN_DIAGRAM,"GRID_DIAGRAM":GRID_DIAGRAM,
        "SMALL_DIAGRAM":entity_diagram(16,32,2),"BASE_DIAGRAM":entity_diagram(32,64,3),"TEACHER_DIAGRAM":TEACHER_DIAGRAM,
        "ORIGINAL_FEATURES":table(["Channel","Feature","Encoding"],original),
        "GRID_FEATURES":table(["Channel","Feature","Encoding"],grid),
        **{key:table(["Index","Feature","Encoding"],value) for key,value in zip(["CELL_FEATURES","SNAKE_FEATURES","FOOD_FEATURES","GLOBAL_FEATURES"],features)},
        "E001_CONFIG":table(["Run","Mini-batch b","Rollout R","Environments","R/b"],[[k,f"{base[k].batch_size:,}",f"{base[k].rollout_samples:,}",f"{base[k].num_envs:,}",f"{base[k].rollout_samples/base[k].batch_size:g}"] for k in "ABCDEF"]),
        "E001_RESULTS":table(["Run","Eval @300M","Eval @600M","Best score","Last score","Last 4 evals","Training M"],[[k,number(base[k].eval_at_300M),number(base[k].eval_at_600M),number(base[k].best_eval),number(base[k].last_eval),number(base[k].last4_eval_mean),millions(base[k].last_logged_samples)] for k in "ABCDEF"]),
        "E002_EARLY":table(["Scale","Epochs","Score @25M","Score @50M","Score @100M"],early),
        "E002_RESULTS":table(["Scale","Epochs","Score @300M","Best score","Last score","Last 4 evals","Training M"],late),
        "E003_RESULTS":table(["Schedule","Score at switch","After +300M","Best score","Last score","Total / additional M"],switches),
        "E004_CONFIG":table(["Run","Model","Raw features","Input projection","Depth","Token dim","Parameters","Teacher KL"],model_config),
        "E004_RESULTS":table(["Run","Score @300M","Best score","Last score","Last 4 evals","Training / last-eval M"],architecture_results),
        "E004_SHARED":table(["Model","Score @300M","Score @400M","Score @575M"],[[name,*[number(at(r,t)) if r.last_eval_samples>=t*1e6 else "—" for t in [300,400,575]]] for name,r in [("Baseline MLP",base["C"]),("Small",lookup[("E004","B-v7")]),("Base",lookup[("E004","C")]),("Small + teacher KL",lookup[("E004","D")])]]),
    }
