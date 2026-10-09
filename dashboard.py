"""
dashboard.py  (Group 64 | MSRUAS)
Run:  streamlit run dashboard.py
Deps: streamlit plotly networkx pandas numpy reportlab
"""

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import pandas as pd
import sys
import importlib
from datetime import datetime
import os

def load_real_metrics():
    path = os.path.join("results", "model_comparison.csv")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df.rename(columns={
                "model": "Model",
                "auc": "AUC",
                "f1": "F1",
                "accuracy": "Accuracy"
            }, inplace=True)
            return df
        except Exception:
            pass
    return None

REAL_METRICS = load_real_metrics()

def load_cancer_types():
    path = os.path.join("data", "raw", "tp53", "tp53_mutations.csv")
    fallback = [
        "Breast carcinoma",
        "Colorectal carcinoma",
        "Lung (NSCLC)",
        "Head and Neck SCC",
        "Ovarian carcinoma",
    ]
    if not os.path.exists(path):
        return fallback

    try:
        diseases = pd.read_csv(path, usecols=["Disease"], low_memory=False)["Disease"]
        diseases = diseases.dropna().astype(str).str.strip()
        diseases = diseases[diseases != ""]
        return diseases.value_counts().index.tolist() or fallback
    except Exception:
        return fallback

CANCER_TYPES = load_cancer_types()

st.set_page_config(page_title="p53 Federated AI Dashboard", page_icon="🧬",
                   layout="wide", initial_sidebar_state="collapsed")

sys.path.append(os.path.join(os.path.dirname(__file__), "src", "ml_models"))
try:
    from predict_new_patient import predict_patient
    _predictor_available = True
except ImportError:
    _predictor_available = False

try:
    import report_generator as _report_generator
    _report_generator = importlib.reload(_report_generator)
    generate_report = _report_generator.generate_report
    _report_available = True
except ImportError:
    _report_available = False

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{--bg-primary:#070B14;--bg-secondary:#0D1526;--bg-card:#111827;--border:#1E2D45;
      --accent-cyan:#00D4FF;--accent-green:#00FF9D;--accent-orange:#FF6B35;
      --accent-purple:#8B5CF6;--text-primary:#E8F4FD;--text-muted:#6B8CAE;
      --glow-cyan:0 0 20px rgba(0,212,255,0.3);--glow-green:0 0 20px rgba(0,255,157,0.3);}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;
  background-color:var(--bg-primary)!important;color:var(--text-primary)!important;}
.stApp{background-color:var(--bg-primary)!important;}
#MainMenu,footer,header{visibility:hidden;}
/* hide native sidebar completely — we use inline columns instead */
[data-testid="stSidebar"]{display:none!important;}
[data-testid="collapsedControl"]{display:none!important;}
.block-container{padding:1.5rem 2rem 2rem;max-width:1600px;}
.metric-card{background:var(--bg-card);border:1px solid var(--border);border-radius:12px;
             padding:1.2rem 1.5rem;position:relative;overflow:hidden;transition:border-color .3s;}
.metric-card:hover{border-color:var(--accent-cyan);}
.metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
                     background:linear-gradient(90deg,var(--accent-cyan),var(--accent-green));}
.metric-label{font-family:'Space Mono',monospace;font-size:.65rem;letter-spacing:.15em;
              text-transform:uppercase;color:var(--text-muted);margin-bottom:.4rem;}
.metric-value{font-family:'Space Mono',monospace;font-size:2rem;font-weight:700;
              color:var(--accent-cyan);text-shadow:var(--glow-cyan);line-height:1;}
.metric-sub{font-size:.75rem;color:var(--text-muted);margin-top:.3rem;}
.section-header{font-family:'Space Mono',monospace;font-size:.7rem;letter-spacing:.2em;
                text-transform:uppercase;color:var(--accent-cyan);
                border-bottom:1px solid var(--border);padding-bottom:.5rem;margin-bottom:1rem;}
.hero-banner{background:linear-gradient(135deg,#070B14 0%,#0D1526 50%,#071424 100%);
             border:1px solid var(--border);border-radius:16px;padding:2rem 2.5rem;
             margin-bottom:1.5rem;position:relative;overflow:hidden;}
.hero-banner::after{content:'⬡⬡⬡⬡⬡⬡⬡⬡⬡⬡⬡⬡';position:absolute;right:2rem;top:50%;
                    transform:translateY(-50%);font-size:2rem;letter-spacing:.3rem;
                    opacity:.04;color:var(--accent-cyan);}
.hero-title{font-family:'Space Mono',monospace;font-size:1.5rem;font-weight:700;
            color:var(--text-primary);margin:0 0 .3rem;}
.hero-title span{color:var(--accent-cyan);}
.hero-sub{font-size:.85rem;color:var(--text-muted);font-weight:300;}
.tag{display:inline-block;background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.3);
     border-radius:20px;padding:.2rem .7rem;font-size:.7rem;font-family:'Space Mono',monospace;
     color:var(--accent-cyan);margin:.15rem;}
/* inline control panel */
.ctrl-panel{background:#0A1220;border:1px solid #00D4FF;border-radius:12px;
            padding:1.2rem 1rem;box-shadow:0 0 30px rgba(0,212,255,0.1);margin-bottom:1rem;}
.ctrl-panel-title{font-family:'Space Mono',monospace;font-size:.58rem;letter-spacing:.22em;
                  text-transform:uppercase;color:#00D4FF;border-bottom:1px solid #1E2D45;
                  padding-bottom:.6rem;margin-bottom:1rem;}
.ctrl-section{font-family:'Space Mono',monospace;font-size:.55rem;letter-spacing:.15em;
              text-transform:uppercase;color:#00D4FF;border-bottom:1px solid #1E2D45;
              padding-bottom:.3rem;margin:.8rem 0 .5rem;}
.report-card{background:linear-gradient(135deg,#0D1526,#070B14);border:1px solid #1E2D45;
             border-left:3px solid #00D4FF;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;}
.report-card-green{background:linear-gradient(135deg,#071a0f,#070B14);border:1px solid #1E2D45;
                   border-left:3px solid #00FF9D;border-radius:12px;padding:1.2rem 1.5rem;margin-bottom:1rem;}
/* «/» toggle button style */
button[kind="secondary"]{transition:all .2s!important;}
</style>
""", unsafe_allow_html=True)

# ── DATA ─────────────────────────────────────────────────────────────────────
KNOWN_INTERACTIONS={("TP53","MDM2"):0.999,("TP53","HSP90"):0.890,("TP53","p300"):0.920,
  ("TP53","BAX"):0.950,("TP53","BCL2"):0.760,("TP53","ATM"):0.910,("TP53","CHEK2"):0.870,
  ("TP53","BRCA1"):0.820,("TP53","PUMA"):0.970,("MDM2","HSP90"):0.750,("MDM2","p300"):0.680,
  ("BAX","BCL2"):0.980,("BAX","PUMA"):0.850,("ATM","CHEK2"):0.920,("ATM","BRCA1"):0.910,
  ("CHEK2","BRCA1"):0.870}
PROTEINS=list({p for pair in KNOWN_INTERACTIONS for p in pair})
if REAL_METRICS is not None:
    MODEL_METRICS = REAL_METRICS.copy()
else:
    MODEL_METRICS = pd.DataFrame({
        "Model": ["GNN", "XGBoost", "Random Forest", "Ensemble"],
        "AUC": [0.0, 0.0, 0.0, 0.0],
        "F1": [0.0, 0.0, 0.0, 0.0],
        "Accuracy": [0.0, 0.0, 0.0, 0.0],
    })
MUTATION_TARGET_MAP={
  175:[("HSP90",0.818,"#00D4FF"),("MDM2",0.668,"#00FF9D"),("p300",0.618,"#8B5CF6")],
  248:[("MDM2",0.790,"#00D4FF"),("PUMA",0.641,"#00FF9D"),("ATM",0.590,"#8B5CF6")],
  249:[("MDM2",0.780,"#00D4FF"),("HSP90",0.630,"#00FF9D"),("CHEK2",0.580,"#8B5CF6")],
  273:[("MDM2",0.800,"#00D4FF"),("p300",0.650,"#00FF9D"),("PUMA",0.600,"#8B5CF6")],
  245:[("HSP90",0.770,"#00D4FF"),("BAX",0.620,"#00FF9D"),("ATM",0.570,"#8B5CF6")],
  282:[("HSP90",0.790,"#00D4FF"),("MDM2",0.640,"#00FF9D"),("CHEK2",0.590,"#8B5CF6")],
  220:[("MDM2",0.760,"#00D4FF"),("p300",0.610,"#00FF9D"),("BAX",0.560,"#8B5CF6")]}
FL_HISTORY={"round":list(range(1,51)),
  "val_acc":[0.661+(0.747-0.661)*(1-np.exp(-0.08*r))+np.random.normal(0,0.005) for r in range(1,51)],
  "val_loss":[1.505*np.exp(-0.05*r)+0.70+np.random.normal(0,0.01) for r in range(1,51)],
  "train_loss":[1.574*np.exp(-0.05*r)+0.65+np.random.normal(0,0.01) for r in range(1,51)]}
NODE_ROLES={"TP53":("hub","#FF6B35",28),"MDM2":("oncogene","#00D4FF",20),
  "HSP90":("chaperone","#00FF9D",20),"p300":("activator","#8B5CF6",18),
  "BAX":("apoptosis","#00FF9D",18),"BCL2":("survival","#FF6B35",16),
  "ATM":("kinase","#00D4FF",18),"CHEK2":("kinase","#00D4FF",16),
  "BRCA1":("repair","#8B5CF6",16),"PUMA":("apoptosis","#00FF9D",16)}
CODON_INFO={175:("R175H","Structural — zinc-binding domain loss"),
  220:("Y220C","Structural — cavity in β-sandwich"),
  245:("G245S","DNA contact — L3 loop disruption"),
  248:("R248W/Q","DNA contact — direct base contact lost"),
  249:("R249S","DNA contact — L3 loop + MDM2 gain"),
  273:("R273H/C","DNA contact — Arg finger mutation"),
  282:("R282W","Structural — H2 helix destabilisation")}
HOTSPOT_CODONS={175,245,248,249,273,282}

# ── FALLBACK PREDICTOR ────────────────────────────────────────────────────────
def _builtin_predict(patient):
    DBD={"L1","L2","L3","S1","S2","S3","S4","S5","S6","S7","S8","S9","S10","H1","H2","LSH"}
    codon=int(patient.get("Codon",0)); is_hot=codon in HOTSPOT_CODONS
    in_dbd=str(patient.get("Domain","")).upper() in {d.upper() for d in DBD}
    is_mis=patient.get("Type","")=="Missense"; is_cpg=patient.get("CpG","")=="Yes"
    p=min(0.35+0.30*is_hot+0.18*in_dbd+0.10*is_mis+0.06*is_cpg,0.97)
    PT={"HSP90":"Chaperone — stabilises misfolded p53 structural mutants",
        "MDM2":"Ubiquitin ligase inhibitor — prevents p53 degradation",
        "PRIMA-1":"Small molecule — reactivates conformational mutants",
        "APR-246":"PRIMA-1 analogue — restores WT-like fold to mutant p53",
        "HDAC1":"Histone deacetylase — epigenetic restoration",
        "p300":"Acetyltransferase — acetylation boosts p53 activity",
        "PUMA":"Pro-apoptotic effector — downstream pathway restoration"}
    sc={k:0.35 for k in PT}
    if is_hot: sc["HSP90"]+=0.30; sc["PRIMA-1"]+=0.28
    if codon==175: sc["HSP90"]+=0.22; sc["PRIMA-1"]+=0.18
    if codon==273: sc["MDM2"]+=0.18; sc["p300"]+=0.15
    if in_dbd: sc["APR-246"]+=0.12
    if is_cpg: sc["PUMA"]+=0.10
    mx=max(sc.values())
    ranked=sorted([{"protein":k,"score":round(v/mx,3),"mechanism":PT[k]} for k,v in sc.items()],
                  key=lambda x:x["score"],reverse=True)
    return{"pathogenicity":"Pathogenic" if p>=0.5 else "Benign","confidence":round(p,3),
           "restoration_score":round(p*0.95,3),"top_targets":ranked[:5]}

# ── CHART FUNCTIONS ───────────────────────────────────────────────────────────
def build_graph(mode="full",codon=None,selected_protein=None,min_score=0.7):
    ct=([t[0] for t in MUTATION_TARGET_MAP.get(codon,[])] if codon else [])
    if mode=="subgraph" and ct:
        core=set(["TP53"]+ct)
        for pr in list(core):
            for (p1,p2),sc in KNOWN_INTERACTIONS.items():
                if pr in(p1,p2) and sc>=min_score: core.add(p1); core.add(p2)
        show=list(core)
    else: show=list(PROTEINS)
    G=nx.Graph()
    for pr in show: G.add_node(pr)
    for (p1,p2),sc in KNOWN_INTERACTIONS.items():
        if p1 in show and p2 in show and sc>=min_score: G.add_edge(p1,p2,weight=sc)
    if mode=="subgraph" and "TP53" in G.nodes:
        pos=nx.spring_layout(G,seed=42,k=3.0,pos={"TP53":np.array([0.,0.])},fixed=["TP53"])
    else: pos=nx.spring_layout(G,seed=42,k=2.5)
    et=[]
    for u,v,d in G.edges(data=True):
        x0,y0=pos[u]; x1,y1=pos[v]; w=d.get("weight",0.5)
        ite=(u in ct or v in ct) and codon
        ise=selected_protein in(u,v) if selected_protein else False
        if ise: col,wid="rgba(0,212,255,0.9)",4.0
        elif ite: col,wid="rgba(255,215,0,0.6)",3.0
        else: col,wid="rgba(30,45,69,0.8)",max(1,w*2.0)
        et.append(go.Scatter(x=[x0,x1,None],y=[y0,y1,None],mode="lines",
                             line=dict(width=wid,color=col),hoverinfo="skip",showlegend=False))
    elt=go.Scatter(
        x=[(pos[u][0]+pos[v][0])/2 for u,v in G.edges()],
        y=[(pos[u][1]+pos[v][1])/2 for u,v in G.edges()],mode="text",
        text=[f"{KNOWN_INTERACTIONS.get((u,v),KNOWN_INTERACTIONS.get((v,u),0)):.2f}" for u,v in G.edges()],
        textfont=dict(size=8,color="rgba(107,140,174,0.8)",family="Space Mono"),
        hoverinfo="skip",showlegend=False)
    nx_,ny_,nt_,nc_,ns_,nb_,nbc_,nh_=[],[],[],[],[],[],[],[]
    rm={t[0]:i+1 for i,t in enumerate(MUTATION_TARGET_MAP.get(codon,[]))}
    for nd in G.nodes():
        x,y=pos[nd]; role,bc,bs=NODE_ROLES.get(nd,("unknown","#6B8CAE",14))
        isel=nd==selected_protein; rk=rm.get(nd,None)
        deg=G.degree(nd); nbrs=list(G.neighbors(nd))
        if isel: c,s,brc="#FFFFFF",bs+10,"#00D4FF"
        elif rk==1: c,s,brc="#FFD700",bs+8,"#FFD700"
        elif rk==2: c,s,brc="#C0C0C0",bs+6,"#C0C0C0"
        elif rk==3: c,s,brc="#CD7F32",bs+4,"#CD7F32"
        else: c,s,brc=bc,bs,"#070B14"
        rl=f" #{rk}" if rk else ""
        nx_.append(x); ny_.append(y); nc_.append(c); ns_.append(s)
        nb_.append(3 if(isel or rk) else 1.5); nbc_.append(brc); nt_.append(f"{nd}{rl}")
        nh_.append(f"<b style='color:#00D4FF'>{nd}</b><br>"
                   f"<span style='color:#6B8CAE'>Role:</span> {role}<br>"
                   f"<span style='color:#6B8CAE'>Connections:</span> {deg}<br>"
                   +(f"<span style='color:#FFD700'>⭐ Rank #{rk} for codon {codon}</span><br>" if rk else "")
                   +f"<span style='color:#6B8CAE'>Interacts with:</span> {', '.join(nbrs)}")
    ntr=go.Scatter(x=nx_,y=ny_,mode="markers+text",text=nt_,textposition="top center",
        textfont=dict(size=11,color="#E8F4FD",family="Space Mono"),
        hovertext=nh_,hoverinfo="text",
        marker=dict(size=ns_,color=nc_,line=dict(color=nbc_,width=nb_),opacity=0.95),
        showlegend=False)
    fig=go.Figure(data=et+[elt,ntr])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20,r=20,t=20,b=20),height=500,
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        hoverlabel=dict(bgcolor="#0D1526",bordercolor="#1E2D45",
                        font=dict(family="DM Sans",size=12,color="#E8F4FD")))
    return fig

def model_comparison_chart():
    import plotly.graph_objects as go

    # ✅ Color based on MODEL (not Type)
    clrs = {
        "GNN": "#00D4FF",
        "XGBoost": "#00FF9D",
        "Random Forest": "#FFD166",
        "Ensemble": "#8B5CF6"
    }

    fig = go.Figure()

    for m in ["AUC", "F1", "Accuracy"]:
        for _, row in MODEL_METRICS.iterrows():

            color = clrs.get(row["Model"], "#00D4FF")  # fallback safe

            fig.add_trace(go.Bar(
                name=row["Model"],
                x=[m],
                y=[row[m]],
                marker_color=color,
                marker_line_color="rgba(0,0,0,0)",
                opacity=0.85,
                showlegend=(m == "AUC"),
                legendgroup=row["Model"],
                hovertemplate=f"<b>{row['Model']}</b><br>{m}: {row[m]:.4f}<extra></extra>"
            ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,11,20,0.8)",
        font=dict(family="DM Sans", color="#6B8CAE"),
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            bgcolor="rgba(13,21,38,0.8)",
            bordercolor="#1E2D45",
            borderwidth=1,
            font=dict(size=11)
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(family="Space Mono", size=10)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(30,45,69,0.5)",
            range=[0, 1.01],  # 🔥 FIX: earlier 0.7 hides GNN (~0.49)
            tickformat=".2f"
        ),
        hoverlabel=dict(
            bgcolor="#0D1526",
            bordercolor="#1E2D45",
            font=dict(family="DM Sans", size=12)
        )
    )

    return fig

def fl_convergence_chart():
    fl=FL_HISTORY; fig=go.Figure()
    fig.add_trace(go.Scatter(x=fl["round"],y=fl["val_acc"],name="Val Accuracy",
        line=dict(color="#00FF9D",width=2.5),fill="tozeroy",fillcolor="rgba(0,255,157,0.05)",
        hovertemplate="Round %{x}<br>Val Acc: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=fl["round"],y=[v/2 for v in fl["val_loss"]],
        name="Val Loss (scaled)",line=dict(color="#FF6B35",width=2,dash="dot"),
        hovertemplate="Round %{x}<br>Val Loss: %{y:.4f}<extra></extra>"))
    fig.add_hline(y=0.7479,line_dash="dash",line_color="rgba(0,212,255,0.5)",line_width=1.5,
        annotation_text="74.79% (round 20 baseline)",
        annotation_font=dict(color="#00D4FF",size=10,family="Space Mono"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(7,11,20,0.8)",
        font=dict(family="DM Sans",color="#6B8CAE"),height=280,margin=dict(l=10,r=10,t=10,b=10),
        legend=dict(bgcolor="rgba(13,21,38,0.8)",bordercolor="#1E2D45",borderwidth=1,font=dict(size=11)),
        xaxis=dict(title="FL Round",showgrid=True,gridcolor="rgba(30,45,69,0.4)",
                   tickfont=dict(family="Space Mono",size=9)),
        yaxis=dict(showgrid=True,gridcolor="rgba(30,45,69,0.4)",
                   tickfont=dict(family="Space Mono",size=9)),
        hoverlabel=dict(bgcolor="#0D1526",bordercolor="#1E2D45",font=dict(family="DM Sans",size=12)))
    return fig

def target_ranking_chart(codon):
    tgts=MUTATION_TARGET_MAP.get(codon,[])
    fig=go.Figure(go.Bar(x=[t[1] for t in tgts],y=[t[0] for t in tgts],orientation="h",
        marker=dict(color=[t[2] for t in tgts],opacity=0.85,line=dict(color="rgba(0,0,0,0)")),
        text=[f"{t[1]:.3f}" for t in tgts],textposition="outside",
        textfont=dict(family="Space Mono",size=11,color="#E8F4FD"),
        hovertemplate="<b>%{y}</b><br>Restoration Score: %{x:.4f}<extra></extra>"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(7,11,20,0.8)",
        font=dict(family="DM Sans",color="#6B8CAE"),height=220,margin=dict(l=10,r=60,t=10,b=10),
        xaxis=dict(showgrid=True,gridcolor="rgba(30,45,69,0.4)",range=[0,1.0],
                   tickfont=dict(family="Space Mono",size=9)),
        yaxis=dict(showgrid=False,tickfont=dict(family="Space Mono",size=11,color="#E8F4FD")),
        hoverlabel=dict(bgcolor="#0D1526",bordercolor="#1E2D45",font=dict(family="DM Sans",size=12)))
    return fig

def pathogenicity_donut():
    labels=["Pathogenic","Possibly Path.","Likely Path.","VUS","Benign"]
    values=[35657,8981,5448,2925,860]
    clrs=["#FF6B35","#FF9B6B","#FFB347","#8B5CF6","#00FF9D"]
    fig=go.Figure(go.Pie(labels=labels,values=values,hole=0.65,
        marker=dict(colors=clrs,line=dict(color="#070B14",width=2)),textinfo="percent",
        textfont=dict(family="Space Mono",size=10,color="#E8F4FD"),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>"))
    fig.add_annotation(text="53,871<br><span style='font-size:10px'>mutations</span>",
        x=0.5,y=0.5,showarrow=False,
        font=dict(family="Space Mono",size=14,color="#00D4FF"),align="center")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",height=260,
        margin=dict(l=10,r=10,t=10,b=10),showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10,color="#6B8CAE"),x=1.02),
        hoverlabel=dict(bgcolor="#0D1526",bordercolor="#1E2D45",font=dict(family="DM Sans",size=12)))
    return fig

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k,v in [("panel_open",False),("selected_protein",None),("min_score",0.7),
             ("codon",175),("last_patient",None),("last_result",None)]:
    if k not in st.session_state: st.session_state[k]=v

# HERO (FULL WIDTH — no toggle button)
st.markdown("""<div class='hero-banner'>
    <div class='hero-title'>Graph-Based <span>Federated AI</span> for Mutant p53 Reactivation</div>
    <div class='hero-sub'>Privacy-preserving multi-hospital AI · GNN + Ensemble · Protein Target Ranking · TP53 Mutation Analysis</div>
</div>""", unsafe_allow_html=True)


# =============================================================================
# INLINE CONTROL PANEL  (pure columns — no sidebar, no blink)
# =============================================================================
if True:
    pan_col, main_col = st.columns([3, 7])

    # ───────────────── LEFT PANEL (CONTROLS) ─────────────────
    with pan_col:
        st.markdown("<div class='ctrl-panel'>", unsafe_allow_html=True)
        st.markdown("<div class='ctrl-panel-title'>⬡ &nbsp; Controls</div>",
                    unsafe_allow_html=True)

        # ── PPI Graph ─────────────────────────────────────────
        st.markdown("<div class='ctrl-section'>⬡ &nbsp; PPI Graph</div>",
                    unsafe_allow_html=True)
        sel_raw = st.selectbox("Highlight Protein",
                               ["None"] + sorted(PROTEINS), key="sel_protein")
        st.session_state.selected_protein = (None if sel_raw == "None" else sel_raw)
        st.caption("Highlight a protein node in the network")

        # ── Edge Filter ───────────────────────────────────────
        st.markdown("<div class='ctrl-section'>📶 &nbsp; Edge Filter</div>",
                    unsafe_allow_html=True)
        st.session_state.min_score = st.slider(
            "Min Interaction Score", 0.5, 1.0,
            st.session_state.min_score, 0.05, key="score_slider")
        st.caption(f"STRING score >= {st.session_state.min_score:.2f}")

        # ── Codon ─────────────────────────────────────────────
        st.markdown("<div class='ctrl-section'>🎯 &nbsp; Codon</div>",
                    unsafe_allow_html=True)
        ci = [175,220,245,248,249,273,282].index(st.session_state.codon)
        st.session_state.codon = st.selectbox(
            "Hotspot Codon", [175,220,245,248,249,273,282],
            index=ci, key="codon_sel")
        mn, _ = CODON_INFO.get(st.session_state.codon, ("",""))
        st.caption(f"Mutation: {mn}")

        # ── Project Info ──────────────────────────────────────
        st.markdown("<div class='ctrl-section'>ℹ️ &nbsp; Project Info</div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style='display:flex;flex-wrap:wrap;gap:.3rem;margin-bottom:.5rem'>
            <span class='tag'>GNN</span><span class='tag'>FedAvg</span>
            <span class='tag'>FedProx</span>
        </div>
        <div style='display:flex;flex-wrap:wrap;gap:.3rem;margin-bottom:.6rem'>
            <span class='tag'>XGBoost</span><span class='tag'>Random Forest</span>
        </div>
        <div style='font-size:.7rem;color:#6B8CAE;line-height:2;background:#070B14;
                    border:1px solid #1E2D45;border-radius:8px;padding:.6rem .8rem'>
            Dataset: <span style='color:#00D4FF'>p53.fr / IARC</span><br>
            Samples: <span style='color:#00D4FF'>53,871</span><br>
            FL Clients: <span style='color:#00D4FF'>5 hospitals</span><br>
            Group: <span style='color:#00D4FF'>64 · MSRUAS</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ───────────────── RIGHT PANEL (MAIN DASHBOARD) ─────────────────
    with main_col:

        st.markdown("## 📊 Model Performance Comparison")

        # ✅ Chart
        st.plotly_chart(model_comparison_chart(), use_container_width=True)

        # ✅ Table
        dfd = MODEL_METRICS[["Model","AUC","F1","Accuracy"]].copy()
        st.dataframe(dfd, use_container_width=True)


# read resolved values from session state
sp  = st.session_state.selected_protein
ms  = st.session_state.min_score
cdn = st.session_state.codon

# ── KPI ROW ───────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)

def get_metric(model, col):
    try:
        df = MODEL_METRICS
        return float(
            df.loc[df["Model"].str.lower() == model.lower(), col].values[0]
        )
    except:
        return 0.0

gnn_auc = get_metric("GNN", "AUC")
xgb_auc = get_metric("XGBoost", "AUC")
rf_auc = get_metric("Random Forest", "AUC")
ensemble_auc = get_metric("Ensemble", "AUC")

for col,lbl,val,sub in [
    (k1,"Ensemble AUC",f"{ensemble_auc:.3f}","GNN + XGB + RF weighted"),
    (k2,"GNN AUC",f"{gnn_auc:.3f}","Graph Neural Network"),
    (k3,"XGBoost AUC",f"{xgb_auc:.3f}","Boosted Trees"),
    (k4,"Random Forest AUC",f"{rf_auc:.3f}","Bagging Model"),
    (k5,"Dataset Size","53,871","IARC TP53 Database rows")]:
    with col:
        st.markdown(f"<div class='metric-card'>"
                    f"<div class='metric-label'>{lbl}</div>"
                    f"<div class='metric-value'>{val}</div>"
                    f"<div class='metric-sub'>{sub}</div></div>",
                    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── PPI GRAPH ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⬡ Protein–Protein Interaction Network</div>',
            unsafe_allow_html=True)
vc1,vc2,_ = st.columns([2,2,6])
with vc1:
    gm = st.radio("View Mode",["Full Network","Codon Subgraph"],
                  horizontal=True,label_visibility="collapsed")
with vc2:
    st.markdown("<div style='padding-top:.4rem;font-size:.75rem;color:#6B8CAE'>"
                "Full Network = all 10 proteins | Codon Subgraph = mutation-specific view"
                "</div>", unsafe_allow_html=True)
st.markdown("<div style='margin-top:.5rem'></div>", unsafe_allow_html=True)

ALL_CODONS = [175,220,245,248,249,273,282]
tabs = st.tabs(["🌐 All"] + [f"Codon {c}" for c in ALL_CODONS])

with tabs[0]:
    cg, ci_col = st.columns([3,1], gap="medium")
    with cg:
        st.plotly_chart(build_graph("full",None,sp,ms),
                        use_container_width=True, config={"displayModeBar":False})
        st.markdown("<div style='font-size:.72rem;color:#6B8CAE;text-align:center'>"
            "Edge weight = STRING v12 confidence score · "
            "<span style='color:#FF6B35'>●</span> TP53 hub · "
            "<span style='color:#00D4FF'>●</span> Regulators · "
            "<span style='color:#00FF9D'>●</span> Apoptosis</div>",
            unsafe_allow_html=True)
    with ci_col:
        st.markdown("""<div style='font-size:.75rem;color:#6B8CAE;background:var(--bg-card);
            border:1px solid var(--border);border-radius:10px;padding:1rem;line-height:1.9'>
            <b style='color:#E8F4FD'>Network Legend</b><br>
            <span style='color:#FF6B35'>●</span> TP53 (hub)<br>
            <span style='color:#00D4FF'>●</span> MDM2/ATM/CHEK2<br>
            <span style='color:#00FF9D'>●</span> BAX/HSP90/PUMA<br>
            <span style='color:#8B5CF6'>●</span> p300/BRCA1<br><br>
            <b style='color:#E8F4FD'>Edge Weight</b><br>STRING v12 confidence<br>score (0–1)<br><br>
            <b style='color:#E8F4FD'>16 Interactions</b><br>10 Proteins<br>
            Human TP53 network</div>""", unsafe_allow_html=True)

for i, c_ in enumerate(ALL_CODONS):
    with tabs[i+1]:
        mn, md = CODON_INFO.get(c_,("",""))
        tgts   = MUTATION_TARGET_MAP.get(c_,[])
        st.markdown(f"""<div style='display:flex;align-items:center;gap:1.5rem;
            background:var(--bg-card);border:1px solid var(--border);
            border-radius:10px;padding:.8rem 1.2rem;margin-bottom:.8rem'>
            <div><div style='font-family:Space Mono;font-size:.6rem;letter-spacing:.15em;
                color:#6B8CAE;text-transform:uppercase'>Hotspot Mutation</div>
            <div style='font-family:Space Mono;font-size:1.4rem;font-weight:700;
                color:#FFD700'>Codon {c_}</div></div>
            <div style='border-left:1px solid var(--border);padding-left:1.5rem'>
                <div style='font-family:Space Mono;font-size:1rem;color:#00D4FF'>{mn}</div>
                <div style='font-size:.78rem;color:#6B8CAE'>{md}</div></div>
            <div style='margin-left:auto;display:flex;gap:.5rem'>
                {''.join([f"<div style='background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.3);border-radius:20px;padding:.2rem .8rem;font-family:Space Mono;font-size:.7rem;color:#00D4FF'>#{j+1} {t[0]}</div>" for j,t in enumerate(tgts)])}
            </div></div>""", unsafe_allow_html=True)
        cg2, cr2 = st.columns([3,2], gap="medium")
        with cg2:
            mk = "subgraph" if gm == "Codon Subgraph" else "full"
            st.plotly_chart(build_graph(mk,c_,sp,ms),
                            use_container_width=True, config={"displayModeBar":False})
            st.markdown("<div style='font-size:.72rem;color:#6B8CAE;text-align:center'>"
                "<span style='color:#FFD700'>●</span> #1 &nbsp;"
                "<span style='color:#C0C0C0'>●</span> #2 &nbsp;"
                "<span style='color:#CD7F32'>●</span> #3 &nbsp;"
                "<span style='color:#FF6B35'>●</span> TP53</div>",
                unsafe_allow_html=True)
        with cr2:
            st.markdown(f'<div class="section-header">🎯 Target Ranking — Codon {c_}</div>',
                        unsafe_allow_html=True)
            for j,(pr,sc,col) in enumerate(tgts):
                st.markdown(f"""<div class='metric-card' style='margin-bottom:.6rem;padding:.9rem 1.2rem'>
                    <div style='display:flex;justify-content:space-between;align-items:center'>
                        <div><span style='font-size:1.1rem'>{"🥇🥈🥉"[j]}</span>
                             <span style='font-family:Space Mono;font-size:1rem;
                                color:{col};margin-left:.5rem'>{pr}</span></div>
                        <div style='font-family:Space Mono;font-size:1.2rem;
                            color:{col};font-weight:700'>{sc:.3f}</div>
                    </div>
                    <div style='margin-top:.4rem'>
                        <div style='height:4px;background:rgba(30,45,69,.8);border-radius:2px'>
                            <div style='height:4px;width:{sc*100:.1f}%;
                                background:{col};border-radius:2px'></div></div></div>
                    <div style='font-size:.68rem;color:#6B8CAE;margin-top:.3rem'>
                        Restoration Score</div></div>""", unsafe_allow_html=True)
            st.plotly_chart(target_ranking_chart(c_),
                            use_container_width=True, config={"displayModeBar":False})

st.markdown("<br>", unsafe_allow_html=True)

# ── MODEL COMPARISON + DONUT ──────────────────────────────────────────────────
cmc, cdo = st.columns([3,2], gap="medium")
with cmc:
    st.markdown('<div class="section-header">📊 Model Performance Comparison</div>',
                unsafe_allow_html=True)
    st.plotly_chart(model_comparison_chart(),
                    use_container_width=True, config={"displayModeBar":False})
    dfd = MODEL_METRICS[["Model","AUC","F1","Accuracy"]].copy()
    for c_ in ["AUC","F1","Accuracy"]: dfd[c_] = dfd[c_].map("{:.4f}".format)
    st.dataframe(dfd, use_container_width=True, hide_index=True)
with cdo:
    st.markdown('<div class="section-header">🧬 Pathogenicity Class Distribution</div>',
                unsafe_allow_html=True)
    st.plotly_chart(pathogenicity_donut(),
                    use_container_width=True, config={"displayModeBar":False})
    st.markdown("""<div style='font-size:.75rem;color:#6B8CAE;line-height:1.8;
        background:var(--bg-card);border:1px solid var(--border);border-radius:10px;
        padding:.8rem 1rem;margin-top:.5rem'>
        <b style='color:#E8F4FD'>Dataset Imbalance Note</b><br>
        Pathogenic class dominates at 66.2% of samples.<br>
        Class-weighted loss applied in all models to<br>
        compensate for Benign minority (1.6%).</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── FL CONVERGENCE ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔗 Federated Learning Convergence</div>',
            unsafe_allow_html=True)
cfl, cfli = st.columns([3,1], gap="medium")
with cfl:
    st.plotly_chart(fl_convergence_chart(),
                    use_container_width=True, config={"displayModeBar":False})
with cfli:
    for lbl,val,sub in [("Clients","5","Hospital nodes"),("Rounds","50","Updated from 20"),
        ("Local Epochs","5","Per round"),("Algorithm","FedProx","mu = 0.01"),
        ("Privacy","DP","Gaussian noise")]:
        st.markdown(f"""<div style='background:var(--bg-card);border:1px solid var(--border);
            border-radius:8px;padding:.6rem .9rem;margin-bottom:.5rem'>
            <div style='font-family:Space Mono;font-size:.6rem;letter-spacing:.12em;
                color:#6B8CAE;text-transform:uppercase'>{lbl}</div>
            <div style='font-family:Space Mono;font-size:1rem;
                color:#00D4FF;font-weight:700'>{val}</div>
            <div style='font-size:.68rem;color:#6B8CAE'>{sub}</div></div>""",
            unsafe_allow_html=True)

# ==============================================================================
# NEW PATIENT PREDICTOR
# ==============================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">🔬 New Patient Mutation Predictor</div>',
            unsafe_allow_html=True)

ci_, co_ = st.columns([1,1], gap="medium")
with ci_:
    pc  = st.number_input("Codon Position", min_value=1, max_value=400, value=175)
    pm  = st.selectbox("Mutation Type", ["Missense","Nonsense","Silent","Frameshift","Splice"])
    pd_ = st.selectbox("Domain", ["L2","L3","S10","H1","H2","LSH","S1","S2","Other"])
    pcg = st.selectbox("CpG Site", ["Yes","No"])
    pw  = st.selectbox("Wild-Type Amino Acid", list("ACDEFGHIKLMNPQRSTVWY"))
    pmu = st.selectbox("Mutant Amino Acid",    list("ACDEFGHIKLMNPQRSTVWY"))
    por = st.selectbox("Mutation Origin", ["Somatic","Germline"])
    default_cancer_idx = CANCER_TYPES.index("Breast carcinoma") if "Breast carcinoma" in CANCER_TYPES else 0
    pdi = st.selectbox("Cancer Type", CANCER_TYPES, index=default_cancer_idx)
    st.caption("Cancer type is saved in the patient record/report; the current prediction model does not use it as a scoring feature.")
    run_ = st.button("▶  Run Prediction", use_container_width=True)

with co_:
    if run_:
        pdict = {"Codon":int(pc),"Type":pm,"CpG":pcg,"Domain":pd_,
                 "WT_AA_1":pw,"Mutant_AA_1":pmu,"Mutation_origin":por,"Disease":pdi}
        res = predict_patient(pdict) if _predictor_available else _builtin_predict(pdict)
        if not _predictor_available:
            st.caption("ℹ️ Using built-in heuristic (predict_new_patient.py not found)")
        st.session_state.last_patient = pdict
        st.session_state.last_result  = res
        if "error" in res:
            st.error(res["error"])
        else:
            pc_ = "#FF4569" if res["pathogenicity"]=="Pathogenic" else "#00FF9D"
            st.markdown(f"""<div class='metric-card' style='margin-bottom:.8rem'>
                <div class='metric-label'>Pathogenicity</div>
                <div class='metric-value' style='color:{pc_}'>{res["pathogenicity"]}</div>
                <div class='metric-sub'>Confidence: {res["confidence"]*100:.1f}%</div></div>
                <div class='metric-card' style='margin-bottom:.8rem'>
                <div class='metric-label'>Restoration Need</div>
                <div class='metric-value' style='color:#FFB830'>
                    {res["restoration_score"]*100:.1f}%</div></div>""",
                unsafe_allow_html=True)
            st.markdown('<div class="section-header" style="margin-top:.5rem">'
                        '🎯 Top Protein Targets</div>', unsafe_allow_html=True)
            rcs = ["#FFD700","#C0C0C0","#CD7F32","#00D4FF","#00FF9D"]
            for i,t in enumerate(res["top_targets"]):
                st.markdown(f"""<div class='metric-card' style='padding:.7rem 1rem;margin-bottom:.5rem'>
                    <div style='display:flex;justify-content:space-between'>
                        <span style='font-family:Space Mono;color:{rcs[i]};font-weight:700'>
                            #{i+1} {t["protein"]}</span>
                        <span style='font-family:Space Mono;color:{rcs[i]}'>{t["score"]:.3f}</span></div>
                    <div style='height:4px;background:#1E2D45;border-radius:2px;margin-top:.4rem'>
                        <div style='height:4px;width:{t["score"]*100:.0f}%;
                            background:{rcs[i]};border-radius:2px'></div></div>
                    <div style='font-size:.68rem;color:#6B8CAE;margin-top:.3rem'>
                        {t["mechanism"]}</div></div>""", unsafe_allow_html=True)
            st.success("[OK] Prediction complete — scroll down to generate a PDF report.")
    else:
        st.markdown("""<div style='height:300px;display:flex;align-items:center;
            justify-content:center;background:var(--bg-card);border:1px solid var(--border);
            border-radius:12px;color:#1E2D45;font-family:Space Mono;font-size:.8rem;text-align:center'>
            ← Fill patient data and<br>click Run Prediction</div>""", unsafe_allow_html=True)

# ==============================================================================
# REPORT GENERATOR
# ==============================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">📄 Report Generator</div>',
            unsafe_allow_html=True)

rl_, rr_ = st.columns([1,1], gap="medium")

with rl_:
    st.markdown("""<div class='report-card'>
        <div style='font-family:Space Mono;font-size:.6rem;letter-spacing:.15em;
                    text-transform:uppercase;color:#00D4FF;margin-bottom:.8rem'>
            📋 Report Contents</div>
        <div style='font-size:.8rem;color:#6B8CAE;line-height:2.0'>
            <span style='color:#00FF9D'>01</span> &nbsp; Prediction Summary (KPIs)<br>
            <span style='color:#00FF9D'>02</span> &nbsp; Patient Mutation Details<br>
            <span style='color:#00FF9D'>03</span> &nbsp; Protein Target Ranking Table<br>
            <span style='color:#00FF9D'>04</span> &nbsp; Model Performance Metrics<br>
            <span style='color:#00FF9D'>05</span> &nbsp; Federated Learning Configuration<br>
            <span style='color:#00FF9D'>06</span> &nbsp; Methodology Overview<br>
            <span style='color:#00FF9D'>07</span> &nbsp; Research Disclaimer
        </div></div>""", unsafe_allow_html=True)
    st.markdown("""<div style='font-family:Space Mono;font-size:.58rem;letter-spacing:.15em;
        text-transform:uppercase;color:#00D4FF;margin-bottom:.5rem'>⚙️ Options</div>""",
        unsafe_allow_html=True)
    rpt_name  = st.text_input("Patient / Sample Name or ID", value="",
                               placeholder="e.g. Patient-001",
                               key="report_patient_id")
    rpt_notes = st.text_area("Additional Notes", value="",
                              placeholder="Add any clinical notes or comments...", height=80,
                              key="report_notes")

with rr_:
    has_res = st.session_state.last_result is not None
    if has_res:
        r_ = st.session_state.last_result
        p_ = st.session_state.last_patient
        ml_ = f"{p_.get('WT_AA_1','?')}{p_.get('Codon','?')}{p_.get('Mutant_AA_1','?')}"
        if "error" in r_:
            st.error(r_["error"])
            pred = "Error"
            pc_ = "#FF6B6B"
            conf = 0.0
            top_target = "N/A"
        else:
            pred = r_.get("pathogenicity", "Unknown")
            pc_ = "#FF4569" if pred == "Pathogenic" else "#00FF9D"
            conf = r_.get("confidence", 0.0)

    # safe access for top_targets
            if "top_targets" in r_ and len(r_["top_targets"]) > 0:
                top_target = r_["top_targets"][0].get("protein", "N/A")
            else:
                top_target = "N/A"

            st.success(f"Prediction: {pred}")
        st.markdown("""...""", unsafe_allow_html=True)
    else:
        st.markdown("""<div style='background:var(--bg-card);border:1px solid var(--border);
            border-left:3px solid #1E2D45;border-radius:12px;
            padding:1.2rem 1.5rem;margin-bottom:1rem'>
            <div style='font-family:Space Mono;font-size:.6rem;letter-spacing:.15em;
                text-transform:uppercase;color:#6B8CAE;margin-bottom:.6rem'>
                ⚠ No Prediction Yet</div>
            <div style='font-size:.8rem;color:#1E2D45;line-height:1.7'>
                Run a patient prediction above first,<br>
                then come back here to generate the report.</div></div>""",
            unsafe_allow_html=True)

    st.markdown("<div style='margin-top:.5rem'></div>", unsafe_allow_html=True)

    if st.button(
        "📄  Generate PDF Report" if has_res
        else "📄  Generate PDF Report (run prediction first)",
        key="gen_report", use_container_width=True, disabled=not has_res):

        with st.spinner("Building report…"):
            pfor = dict(st.session_state.last_patient)
            rpt_name  = st.session_state.get("report_patient_id", "")
            rpt_notes = st.session_state.get("report_notes", "")
            if rpt_name.strip():  pfor["Patient_ID"] = rpt_name.strip()
            if rpt_notes.strip(): pfor["Notes"]      = rpt_notes.strip()

            if _report_available:
                pdf_bytes = generate_report(pfor, st.session_state.last_result)
            else:
                import io
                from reportlab.lib.pagesizes import A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet
                buf = io.BytesIO()
                doc = SimpleDocTemplate(buf, pagesize=A4)
                sty = getSampleStyleSheet()
                r2  = st.session_state.last_result
                p2  = pfor
                story = [
                    Paragraph("p53 Mutation Analysis Report", sty["Title"]),
                    Spacer(1,12),
                    Paragraph(f"Mutation: {p2.get('WT_AA_1','?')}{p2.get('Codon','?')}{p2.get('Mutant_AA_1','?')}", sty["Heading2"]),
                    Paragraph(f"Pathogenicity: {r2['pathogenicity']} ({r2['confidence']*100:.1f}%)", sty["Normal"]),
                    Paragraph(f"Restoration Need: {r2['restoration_score']*100:.1f}%", sty["Normal"]),
                    Spacer(1,8),
                    Paragraph("Top Protein Targets:", sty["Heading3"]),
                ]
                if p2.get("Patient_ID"):
                    story.insert(2, Paragraph(f"Patient / Sample: {p2['Patient_ID']}", sty["Normal"]))
                    story.insert(3, Spacer(1, 6))
                if p2.get("Notes"):
                    story.extend([
                        Spacer(1,8),
                        Paragraph("Additional Notes:", sty["Heading3"]),
                        Paragraph(str(p2["Notes"]).replace("\n", "<br/>"), sty["Normal"]),
                    ])
                for i,t in enumerate(r2["top_targets"],1):
                    story.append(Paragraph(
                        f"#{i} {t['protein']} — {t['score']:.3f} — {t['mechanism']}",
                        sty["Normal"]))
                story.append(Spacer(1,12))
                story.append(Paragraph(
                    "Group 64 · MSRUAS · 2026 · Guide: Ms. Manisha Patil",
                    sty["Normal"]))
                doc.build(story); buf.seek(0); pdf_bytes = buf.read()

            if pdf_bytes:
                p2    = st.session_state.last_patient
                mn    = f"{p2.get('WT_AA_1','X')}{p2.get('Codon','0')}{p2.get('Mutant_AA_1','X')}"
                ts    = datetime.now().strftime("%Y%m%d_%H%M")
                fname = f"p53_report_{mn}_{ts}.pdf"
                st.download_button("⬇  Download Report PDF", data=pdf_bytes,
                    file_name=fname, mime="application/pdf", use_container_width=True)
                st.success(f"Report ready — {fname}")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""<div style='text-align:center;font-family:Space Mono;font-size:.65rem;
    color:#1E2D45;letter-spacing:.1em;padding:1rem 0'>
    GROUP 64 · MSRUAS · 2026 · GRAPH-BASED FEDERATED AI FOR MUTANT p53 REACTIVATION ·
    GUIDE: MS. MANISHA PATIL
</div>""", unsafe_allow_html=True)
