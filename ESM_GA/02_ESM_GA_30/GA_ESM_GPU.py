import pandas as pd
import numpy as np
import random
import torch
from transformers import AutoTokenizer,EsmForMaskedLM
import os

SEED=42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


# ================= device =================
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)
if device.type=="cuda":
    torch.backends.cuda.matmul.allow_tf32=True

# ================= load data =================
pool=pd.read_csv("../mutation_pool_all.csv")
compatibility=np.load("../compatibility.npy")
positions=pool.Position.values.astype(int)
wtAA=pool.WT.values
mutAA=pool.Mut.values
proxy=pool.ProxyScore.values
N=len(pool)
proxy_min,proxy_max=proxy.min(),proxy.max()
ESM_MIN,ESM_MAX=-3.0,6.0

print("\nProxy statistics")
print("mean:",proxy.mean())
print("std:",proxy.std())
print("min:",proxy_min)
print("max:",proxy_max)

# ================= WT sequence =================
wt_seq="MFLRREFGAVAALSVLAHAAPAPAPMQRRDISSTVLDNIDLFAQYSAAAYCSSNIESTGTTLTCDVGNCPLVEAAGATTIDEFDDTSSYGDPTGFIAVDPTNELIVLSFRGSSDLSNWIADLNFGLTSVSSICDGCEMHKGFYEAWEVIADTITSKVEAAVSSYPDYTLVFTGHSYGAALAAVAATVLRNAGYTLDLYNFGQPRIGNLALADYITGQNMGSNYRVTHTDDIVPKLPPELLGYHHFSPEYWITSGNDVTVTTSDVTEVVGVDSTAGNDGTLLDSTTAHRWYTIYISECS"

# ================= load ESM1v =================
model_dir="../esm1v"
if not os.path.exists(model_dir):
    raise FileNotFoundError(model_dir)

tokenizer=AutoTokenizer.from_pretrained(model_dir)
model=EsmForMaskedLM.from_pretrained(model_dir).to(device)

if device.type=="cuda":
    model=model.half()

model.eval()
print("\n✓ ESM1v loaded")

# ================= cache =================
esm_cache={}
fitness_cache={}

# ================= sequence =================
def build_sequence(ind):
    seq=list(wt_seq)
    for idx in ind:
        seq[positions[idx]-1]=mutAA[idx]
    return "".join(seq)

# ================= ESM score =================
def esm_multi_score(ind):

    key=tuple(sorted(ind))
    if key in esm_cache:
        return esm_cache[key]

    mutated_seq=build_sequence(ind)

    masked_sequences=[]
    mut_ids=[]
    wt_ids=[]
    token_positions=[]

    for idx in ind:
        pos=positions[idx]-1
        seq=list(mutated_seq)
        seq[pos]=tokenizer.mask_token

        masked_sequences.append("".join(seq))
        mut_ids.append(tokenizer.convert_tokens_to_ids(mutAA[idx]))
        wt_ids.append(tokenizer.convert_tokens_to_ids(wtAA[idx]))
        token_positions.append(pos+1)

    inputs=tokenizer(
        masked_sequences,
        return_tensors="pt",
        padding=True
    )

    inputs={k:v.to(device) for k,v in inputs.items()}

    with torch.no_grad():
        if device.type=="cuda":
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                logits=model(**inputs).logits
        else:
            logits=model(**inputs).logits

    log_probs=torch.log_softmax(logits.float(),dim=-1)

    scores=[]

    for i in range(len(ind)):
        p=token_positions[i]
        delta=(
            log_probs[i,p,mut_ids[i]]
            -
            log_probs[i,p,wt_ids[i]]
        )
        scores.append(delta.item())

    score=np.mean(scores)
    esm_cache[key]=score

    return score

# ================= fitness =================
def fitness(ind):

    key=tuple(sorted(ind))
    if key in fitness_cache:
        return fitness_cache[key]

    proxy_score=np.mean(proxy[ind])
    esm_score=esm_multi_score(ind)

    proxy_norm=(proxy_score-proxy_min)/(proxy_max-proxy_min+1e-8)

    esm_score=np.clip(
        esm_score,
        ESM_MIN,
        ESM_MAX
    )

    esm_norm=(esm_score-ESM_MIN)/(ESM_MAX-ESM_MIN)

    fit=0.3*proxy_norm+0.7*esm_norm

    fitness_cache[key]=fit
    return fit

# ================= GA parameters =================
MUTATION_NUM=30
POP_SIZE=300
GENERATIONS=200
ELITE=30
MUTATION_RATE=0.3

# ================= initialize =================
def create_individual():

    ind=[]

    while len(ind)<MUTATION_NUM:

        idx=np.random.choice(
            N,
            p=proxy/proxy.sum()
        )

        if idx in ind:
            continue

        if all(
            compatibility[idx,x]!=0
            for x in ind
        ):
            ind.append(idx)

    return ind


def repair(ind):
    child=list(dict.fromkeys(ind))
    while len(child)<MUTATION_NUM:
        idx=np.random.randint(N)
        if idx in child:
            continue
        if all(
            compatibility[idx,x]!=0
            for x in child
        ):
            child.append(idx)
    return child


def crossover(p1,p2):

    merged=list(set(p1+p2))
    random.shuffle(merged)

    child=[]

    for idx in merged:
        if all(
            compatibility[idx,x]!=0
            for x in child
        ):
            child.append(idx)
        if len(child)==MUTATION_NUM:
            break
    return repair(child)


def mutate(ind):

    if random.random()>MUTATION_RATE:
        return ind

    child=ind.copy()
    child.pop(
        random.randint(0,MUTATION_NUM-1)
    )
    return repair(child)


# ================= population =================
population=[
    create_individual()
    for _ in range(POP_SIZE)
]
# ================= convergence record =================
history=[]
generation_top=[]
population_history=[]
# ================= evolution =================
for gen in range(GENERATIONS):

    fits=[]

    print("\n"+"="*60)
    print(f"Generation {gen}")
    print("="*60)

    for i,ind in enumerate(population):
        proxy_score=np.mean(proxy[ind])
        esm_score=esm_multi_score(ind)
        proxy_norm=(
            proxy_score-proxy_min
        )/(proxy_max-proxy_min+1e-8)
        esm_clip=np.clip(
            esm_score,
            ESM_MIN,
            ESM_MAX
        )

        esm_norm=(
            esm_clip-ESM_MIN
        )/(ESM_MAX-ESM_MIN)


        fit=(
            0.3*proxy_norm
            +
            0.7*esm_norm
        )
        muts=[
            f"{positions[idx]}{wtAA[idx]}>{mutAA[idx]}"
            for idx in sorted(ind,key=lambda x:positions[x])
        ]

        population_history.append({
            "Generation": gen,
            "Individual": i+1,
            "Fitness": fit,
            "Proxy": proxy_score,
            "Proxy_norm": proxy_norm,
            "ESM": esm_score,
            "ESM_norm": esm_norm,
            "Mutations":";".join(muts),
            "Sequence":build_sequence(ind)
        })

        fits.append(fit)

        print(
            f"Gen {gen:3d} "
            f"Ind {i+1:3d}/{POP_SIZE} | "
            f"Proxy={proxy_score:.3f} "
            f"(N={proxy_norm:.3f}) | "
            f"ESM={esm_score:.3f} "
            f"(N={esm_norm:.3f}) | "
            f"Fitness={fit:.4f}"
        )
    order=np.argsort(fits)[::-1]

    population=[
        population[i]
        for i in order
    ]
    fits=[
        fits[i]
        for i in order
    ]
    best_proxy=np.mean(proxy[population[0]])
    best_esm=esm_multi_score(population[0])
    best_fit=fits[0]

    mean_fit=np.mean(fits)
    elite_fit=np.mean(fits[:ELITE])      # 或 np.mean(fits[:10])

    muts=[
        f"{positions[idx]}{wtAA[idx]}>{mutAA[idx]}"
        for idx in sorted(population[0],key=lambda x:positions[x])
    ]

    population_df=pd.DataFrame(
    population_history
)

    population_df.to_csv(
        "GA_population_history.csv",
        index=False
)
    print(f"\nGeneration {gen} BEST: Fitness={best_fit:.4f} Proxy={best_proxy:.3f} ESM={best_esm:.3f}")

    best_proxy_norm=(
        best_proxy-proxy_min
    )/(proxy_max-proxy_min+1e-8)

    best_esm_clip=np.clip(
        best_esm,
        ESM_MIN,
        ESM_MAX
    )

    best_esm_norm=(
        best_esm_clip-ESM_MIN
    )/(ESM_MAX-ESM_MIN)

    history.append({
        "Generation":gen,
        "Seed":SEED,
        "Elite_Fitness":elite_fit,
        "Best_Fitness":best_fit,
        "Mean_Fitness":mean_fit,
        "Best_Proxy":best_proxy,
        "Best_Proxy_norm":best_proxy_norm,
        "Best_ESM":best_esm,
        "Best_ESM_norm":best_esm_norm,
        "ESM_cache":len(esm_cache),
        "Fitness_cache":len(fitness_cache)

    })

    history_df=pd.DataFrame(history)

    history_df.to_csv(
        "GA_history.csv",
        index=False
    )


    generation_top.append({
        "Generation":gen,
        "Fitness":best_fit,
        "Elite_Fitness":elite_fit,
        "Proxy":best_proxy,
        "Proxy_norm":best_proxy_norm,
        "ESM":best_esm,
        "ESM_norm":best_esm_norm,
        "Mutations":";".join(muts)
    })

    pd.DataFrame(generation_top).to_csv(
        "GA_generation_top.csv",
        index=False
    )

    next_pop = population[:ELITE]

    # 记录已经存在的个体
    seen = {tuple(sorted(ind)) for ind in next_pop}

    retry = 0

    while len(next_pop) < POP_SIZE:

        p1 = random.choice(population[:100])
        p2 = random.choice(population[:100])

        child = mutate(
            crossover(p1, p2)
        )

        key = tuple(sorted(child))

        if key in seen:

            retry += 1

            # 连续100次都生成重复个体，则随机生成一个新的合法个体
            if retry > 100:

                child = create_individual()
                key = tuple(sorted(child))
                retry = 0

                if key in seen:
                    continue

            else:
                continue

        retry = 0

        seen.add(key)
        next_pop.append(child)

    population = next_pop
    
    print(
        f"\nGeneration {gen} "
        f"Best={fits[0]:.4f} "
        f"ESM_cache={len(esm_cache)} "
        f"Fitness_cache={len(fitness_cache)}\n"
    )
# ================= save Top300 from all generations =================

all_df=pd.DataFrame(population_history)

# 去除重复突变组合，只保留不同variant
all_df=all_df.drop_duplicates(
    subset=["Mutations"]
)

# 按Fitness排序
all_df=all_df.sort_values(
    "Fitness",
    ascending=False
)

# 提取Top300
top300=all_df.head(300).copy()

# 添加Rank
top300.insert(
    0,
    "Rank",
    range(1,len(top300)+1)
)

top300.to_csv(
    "Top300_GA_ESM_AllGenerations.csv",
    index=False
)

print("\nFinished")
print("Total candidates:",len(population_history))
print("Unique variants:",len(all_df))
print(top300.head())
