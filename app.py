from pathlib import Path
p=Path('/mnt/data/app.py')
p.write_text('''"""Interactive Streamlit dashboard; runs the existing model and detectors in-process."""
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from data_generator import generate, TRAIN_END, WINDOW
from drift_engine import DualRunner, StreamEngine, TransferGBR, make_windows

st.set_page_config(page_title="Drift Intelligence | Live Monitor", page_icon="📈", layout="wide")
C = {"static":"#94A3B8","transfer":"#22C55E","actual":"#F8FAFC",
     "alarm":"#FB923C","truth":"#FB7185","adapt":"#2DD4BF",
     "data":"#38BDF8","rel":"#C084FC"}
st.markdown("""<style>
.stApp {background:#0B1220;color:#E2E8F0}
[data-testid="stSidebar"] {background:#111D30}
[data-testid="stMetric"] {background:#17243A;border:1px solid #30445F;
border-radius:12px;padding:14px}
h1,h2,h3 {color:#F1F5F9!important}
div[data-testid="stTabs"] button {font-weight:650}
</style>""", unsafe_allow_html=True)
st.title("◈ Drift Intelligence")
st.caption("LIVE MONITORING  /  KS & PSI DATA DRIFT  /  PAGE-HINKLEY RELATIONAL DRIFT  /  TRANSFER GBR")
@st.cache_resource(show_spinner="Generating data and pre-training model…")
def assets():
    df=generate()
    pretrained=StreamEngine(df, TRAIN_END, WINDOW, TransferGBR, 0).model
    return df, pretrained

def basefig(title, ytitle):
    f=go.Figure()
    f.update_layout(title=title,template="plotly_dark",paper_bgcolor="#111D30",
        plot_bgcolor="#111D30",font_color="#E2E8F0",height=355,
        margin=dict(l=40,r=24,t=60,b=38),xaxis_title="Stream window",
        yaxis_title=ytitle,hovermode="x unified",legend=dict(orientation="h",y=1.13))
    f.update_xaxes(showgrid=True,gridcolor="#24354C")
    f.update_yaxes(showgrid=True,gridcolor="#24354C")
    return f

def marks(f, events, truth=False):
    shown=set()
    for event in events:
        x=event["window"]
        typ=event["type"]
        if typ=="truth" and not truth: continue
        color={"alarm":C["alarm"],"adapt":C["adapt"],"truth":C["truth"]}[typ]
        dash={"alarm":"solid","adapt":"dot","truth":"dash"}[typ]
        label={"alarm":"Detected alarm","adapt":"Re-adaptation","truth":"True regime change"}[typ]
        # legend grouping via invisible markers avoids dozens of repeated legend labels
        if typ not in shown:
            f.add_trace(go.Scatter(x=[None],y=[None],mode="lines",name=label,
                                   line=dict(color=color,dash=dash),showlegend=True))
            shown.add(typ)
        f.add_vline(x=x,line_width=1,line_color=color,line_dash=dash,opacity=.65)
    return f

def optional_csv(name):
    try:
        from pathlib import Path
        path=Path(__file__).parent/name
        return pd.read_csv(path) if path.exists() else None
    except Exception:
        return None

with st.sidebar:
    st.header("Stream controls")
    delay=st.slider("Seconds per window",0.0,2.0,.2,.1)
    show_truth=st.checkbox("Show simulated ground truth",False,
        help="Offline demonstration only; never used by live detection")
    start=st.button("▶ Start / restart stream",type="primary",use_container_width=True)
    st.divider()
    st.caption("Runs entirely inside Streamlit. No separate FastAPI server required.")
    st.caption("Charts only display metrics supplied by the engine or validly computed from its outputs.")

if "rows" not in st.session_state: st.session_state.rows=[]
if "events" not in st.session_state: st.session_state.events=[]
if start:
    st.session_state.rows=[]
    st.session_state.events=[]

tabs=st.tabs(["◉ Live monitor","◈ Drift analysis","▤ Model performance",
              "↗ Adaptation & recovery","◎ Detector evaluation","☷ Event log"])
with tabs[0]:
    kpi=st.columns(5)
    kboxes=[x.empty() for x in kpi]
    main_plot=st.empty()
    st.subheader("Error comparison")
    error_plot=st.empty()
with tabs[1]:
    col1,col2=st.columns(2)
    sensor_plot=col1.empty()
    residual_plot=col2.empty()
    timeline_plot=st.empty()
with tabs[2]:
    st.caption("Only the two strategies emitted by the current DualRunner are plotted live.")
    mae_plot=st.empty()
    extra_metrics=st.empty()
    offline_plot=st.empty()
with tabs[3]:
    recovery_plot=st.empty()
    recovery_table=st.empty()
with tabs[4]:
    st.info("Event-level precision/recall requires a defined ground-truth event-matching tolerance. "
            "This dashboard reports observable alarms and matches without inventing precision/F1.")
    detector_table=st.empty()
with tabs[5]:
    event_table=st.empty()

def draw():
    rows=st.session_state.rows
    events=st.session_state.events
    if not rows: return
    d=pd.DataFrame(rows).sort_values("window")
    m=rows[-1]
    diag=str(m.get("diagnosis","none")).upper()
    kboxes[0].metric("WINDOW",int(m["window"])+1)
    kboxes[1].metric("DIAGNOSIS",diag)
    kboxes[2].metric("STATIC MAE",f'{m["mae_static"]:.2f}')
    kboxes[3].metric("TRANSFER MAE",f'{m["mae"]:.2f}',
        f'{m["mae"]-m["mae_static"]:+.2f} vs static',delta_color="inverse")
    kboxes[4].metric("MODEL TREES",m.get("n_trees","—"))

    # Prediction chart only when the engine actually exposes aligned actual/predicted values.
    actual=next((k for k in ["y_true","actual","target"] if k in d),None)
    pred=next((k for k in ["y_pred","predicted","prediction"] if k in d),None)
    if actual and pred and pd.api.types.is_numeric_dtype(d[actual]) and pd.api.types.is_numeric_dtype(d[pred]):
        f=basefig("Actual vs predicted · live","Target")
        f.add_trace(go.Scatter(x=d.window,y=d[actual],name="Actual",line=dict(color=C["actual"],width=3)))
        f.add_trace(go.Scatter(x=d.window,y=d[pred],name="Prediction",line=dict(color=C["transfer"],width=2)))
        main_plot.plotly_chart(marks(f,events,show_truth),use_container_width=True,key="actual_live")
    else:
        main_plot.info("Actual vs predicted is available once DualRunner.step() emits aligned "
                       "actual and predicted values. Current engine emits window MAE instead.")

    f=basefig("Window MAE · static vs transfer","MAE ↓")
    for key,label,color in [("mae_static","Static",C["static"]),("mae","Transfer learning",C["transfer"])]:
        f.add_trace(go.Scatter(x=d.window,y=d[key],name=label,mode="lines+markers",
                               line=dict(color=color,width=2.5),marker=dict(size=4)))
    error_plot.plotly_chart(marks(f,events,show_truth),use_container_width=True,key="live_mae")

    ks=m.get("ks") or {}
    if ks:
        ss=pd.Series(ks,dtype=float).dropna().sort_values()
        f=go.Figure(go.Bar(x=ss.values,y=ss.index,orientation="h",
            marker_color=[C["alarm"] if x>=.3 else C["data"] for x in ss.values],
            hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        f.update_layout(template="plotly_dark",paper_bgcolor="#111D30",
            plot_bgcolor="#111D30",title="KS statistic · sensors",height=380,
            margin=dict(l=25,r=20,t=50,b=30),xaxis_title="KS statistic")
        sensor_plot.plotly_chart(f,use_container_width=True,key="sensor_ks")
    f=basefig("Drift signals · separate axes","Worst KS")
    f=make_subplots(specs=[[{"secondary_y":True}]])
    f.add_trace(go.Scatter(x=d.window,y=d.worst_ks,name="Worst KS · data",
        line=dict(color=C["data"],width=2)),secondary_y=False)
    f.add_trace(go.Scatter(x=d.window,y=d.mean_resid,name="Normalized residual · relational",
        line=dict(color=C["rel"],width=2)),secondary_y=True)
    f.update_layout(template="plotly_dark",paper_bgcolor="#111D30",plot_bgcolor="#111D30",
        height=380,hovermode="x unified",title="Data vs relational signals",
        legend=dict(orientation="h",y=1.12),margin=dict(l=40,r=45,t=65,b=35))
    f.update_xaxes(title_text="Window",gridcolor="#24354C")
    f.update_yaxes(title_text="KS statistic",secondary_y=False,gridcolor="#24354C")
    f.update_yaxes(title_text="Residual × normal",secondary_y=True,gridcolor="#24354C")
    residual_plot.plotly_chart(marks(f,events,show_truth),use_container_width=True,key="signals")

    f=basefig("Alarm and adaptation timeline","Event")
    event_types={"truth":3,"alarm":2,"adapt":1}
    for e in events:
        if e["type"]=="truth" and not show_truth: continue
        f.add_trace(go.Scatter(x=[e["window"]],y=[event_types[e["type"]]],
            mode="markers",marker=dict(size=13,color={"truth":C["truth"],
                "alarm":C["alarm"],"adapt":C["adapt"]}[e["type"]]),
            name=e["type"].title(),showlegend=False,
            hovertext=e.get("description",""),hoverinfo="x+text"))
    f.update_yaxes(tickvals=[1,2,3],ticktext=["Adaptation","Alarm","True drift"],range=[.5,3.5])
    timeline_plot.plotly_chart(f,use_container_width=True,key="timeline")

    mae_plot.plotly_chart(marks(basefig("Model performance over time · MAE","MAE ↓")
        .add_trace(go.Scatter(x=d.window,y=d.mae_static,name="Static",line=dict(color=C["static"])))
        .add_trace(go.Scatter(x=d.window,y=d.mae,name="Transfer",line=dict(color=C["transfer"]))),
        events,show_truth),use_container_width=True,key="performance_mae")
    optional=[x for x in ["rmse","rmse_static","r2","r2_static"] if x in d]
    if optional:
        extra_metrics.dataframe(d[["window"]+optional].tail(20),hide_index=True)
    else:
        extra_metrics.caption("RMSE/R² are not reconstructible from MAE alone. "
                              "Expose per-window predictions and actuals from the engine to enable them.")
    comparison=optional_csv("model_comparison.csv")
    if comparison is not None:
        offline_plot.subheader("Saved experiment comparison")
        offline_plot.dataframe(comparison,use_container_width=True,hide_index=True)
    else:
        offline_plot.caption("No model_comparison.csv found.")

    # Show observable adaptation markers and recovery, without claiming recovery
    # thresholds or timings the backend has not defined.
    evdf=pd.DataFrame([{"window":e["window"],"event":e["type"],
                       "description":e.get("description","")} for e in events])
    recovery_plot.plotly_chart(marks(basefig("MAE around drift/adaptation","MAE")
        .add_trace(go.Scatter(x=d.window,y=d.mae,name="Transfer",line=dict(color=C["transfer"])))
        .add_trace(go.Scatter(x=d.window,y=d.mae_static,name="Static",line=dict(color=C["static"]))),
        events,show_truth),use_container_width=True,key="recovery")
    if not evdf.empty:
        recovery_table.dataframe(evdf[evdf.event.isin(["alarm","adapt"])],hide_index=True)

    alarms=[e for e in events if e["type"]=="alarm"]
    truth=[e for e in events if e["type"]=="truth"]
    detector_table.metric("Observed drift alarms",len(alarms))
    if show_truth and truth:
        detector_table.caption(f"Observed regime changes: {len(truth)}. "
          "Event-level matching requires an explicit detection-delay tolerance.")
    log=pd.DataFrame([{"window":r["window"],"diagnosis":r.get("diagnosis"),
                       "alarm":r.get("alarm"),"recovery":r.get("recovery"),
                       "shifted sensors":", ".join(r.get("features_moved") or []),
                       **({"true regime":r.get("truth_regime")} if show_truth else {})}
                      for r in rows])
    event_table.dataframe(log.tail(200).iloc[::-1],use_container_width=True,hide_index=True)

if start:
    df,pretrained=assets()
    runner=DualRunner(df,TRAIN_END,WINDOW,pretrained)
    last_truth=None
    progress=st.progress(0,text="Streaming…")
    windows=list(make_windows(len(df),TRAIN_END,WINDOW))
    for i,(s,e) in enumerate(windows):
        msg=runner.step(s,e)
        msg["truth_regime"]=str(df.regime.iloc[s]) if "regime" in df else None
        st.session_state.rows.append(msg)
        current_truth=msg["truth_regime"]
        if current_truth is not None and last_truth is not None and current_truth!=last_truth:
            st.session_state.events.append({"window":msg["window"],"type":"truth",
                                            "description":f"{last_truth} → {current_truth}"})
        last_truth=current_truth
        if msg.get("alarm"):
            st.session_state.events.append({"window":msg["window"],"type":"alarm",
                 "description":str(msg.get("diagnosis",""))})
        if msg.get("recovery"):
            st.session_state.events.append({"window":msg["window"],"type":"adapt",
                 "description":"Drift ended / re-adaptation"})
        # Updating all charts at every window can be expensive on long streams.
        if i%3==0 or i==len(windows)-1:
            draw()
            progress.progress((i+1)/len(windows),text=f"Window {i+1}/{len(windows)}")
        time.sleep(delay)
    progress.empty()
    st.success("Stream finished.")
elif st.session_state.rows:
    draw()
else:
    st.info("Start the live stream from the sidebar. Existing data and ML logic stay unchanged.")
''')
print("Created",p,"bytes",p.stat().st_size)
import ast
ast.parse(p.read_text())
print("Python syntax: PASS")
