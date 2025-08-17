# app.py
# Tablero Intraoperatorio (Congelación) - Streamlit
# Registra hitos, calcula TAT y monitorea SLA.

import sqlite3
from contextlib import closing
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

APP_TITLE = "Tablero intraoperatorio (Congelación)"
DB_PATH_DEFAULT = "intraop.db"
TZ = ZoneInfo("America/Lima")

# ============================
# Persistencia (SQLite)
# ============================
def init_db(db_path: str):
    with closing(sqlite3.connect(db_path)) as con, con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_code TEXT,
            historia_clinica TEXT,
            paciente TEXT,
            servicio TEXT,
            cirujano TEXT,
            pieza TEXT,
            estado TEXT,
            t_inicio TEXT,
            t_recepcion TEXT,
            t_criostato TEXT,
            t_diagnostico TEXT,
            t_comunicado TEXT,
            notas TEXT
        )
        """)
    return db_path

def get_connection(db_path: str):
    return sqlite3.connect(db_path, check_same_thread=False)

@st.cache_data(show_spinner=False)
def load_cases(db_path: str) -> pd.DataFrame:
    with closing(sqlite3.connect(db_path)) as con:
        df = pd.read_sql_query("SELECT * FROM cases ORDER BY id DESC", con)
    return df

def refresh_cache():
    load_cases.clear()

def insert_case(con, **kwargs):
    cols = ",".join(kwargs.keys())
    qs = ",".join(["?"] * len(kwargs))
    with con:
        con.execute(f"INSERT INTO cases ({cols}) VALUES ({qs})", tuple(kwargs.values()))

def update_case_time(con, case_id: int, field: str, dt_iso: str):
    with con:
        con.execute(f"UPDATE cases SET {field}=? WHERE id=?", (dt_iso, case_id))

def update_case_general(con, case_id: int, **kwargs):
    sets = ",".join([f"{k}=?" for k in kwargs.keys()])
    vals = list(kwargs.values()) + [case_id]
    with con:
        con.execute(f"UPDATE cases SET {sets} WHERE id=?", vals)

# ============================
# Métricas y resúmenes
# ============================
def compute_metrics(df: pd.DataFrame, sla_min: int) -> pd.DataFrame:
    """Cálculos vectorizados; tolera DataFrames vacíos y NaT."""
    df = df.copy()
    if df.empty:
        df["min_recep_a_diag"] = pd.Series(dtype="float64")
        df["min_diag_a_com"]   = pd.Series(dtype="float64")
        df["min_total"]        = pd.Series(dtype="float64")
        df["cumple_SLA"]       = pd.Series(dtype="bool")
        df["estado_calc"]      = pd.Series(dtype="object")
        return df

    for col in ["t_recepcion", "t_criostato", "t_diagnostico", "t_comunicado", "t_inicio"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["min_recep_a_diag"] = (df["t_diagnostico"] - df["t_recepcion"]).dt.total_seconds() / 60
    df["min_diag_a_com"]   = (df["t_comunicado"] - df["t_diagnostico"]).dt.total_seconds() / 60
    df["min_total"]        = (df["t_comunicado"] - df["t_recepcion"]).dt.total_seconds() / 60

    df["cumple_SLA"]  = df["min_total"] <= float(sla_min)
    df["estado_calc"] = np.where(df["t_diagnostico"].notna() & df["t_comunicado"].notna(),
                                 "reportado", "pendiente")
    return df

def summarize(df: pd.DataFrame, sla_min: int) -> dict:
    df_ok = df[df["min_total"].notna()]
    if df_ok.empty:
        return {"n": int(len(df)), "n_con_TAT": 0, "mediana_min": np.nan,
                "p90_min": np.nan, "cumplimiento_%": 0.0,
                "prom_recep_diag": np.nan, "prom_diag_com": np.nan}
    return {
        "n": int(len(df)),
        "n_con_TAT": int(len(df_ok)),
        "mediana_min": float(np.nanmedian(df_ok["min_total"])),
        "p90_min": float(np.nanpercentile(df_ok["min_total"], 90)),
        "cumplimiento_%": float(100.0 * (df_ok["min_total"] <= sla_min).mean()),
        "prom_recep_diag": float(np.nanmean(df_ok["min_recep_a_diag"])),
        "prom_diag_com": float(np.nanmean(df_ok["min_diag_a_com"])),
    }

# ============================
# Datos demo
# ============================
def seed_demo(con):
    now = datetime.now(TZ).replace(second=0, microsecond=0)
    ejemplos = []
    for i, tdelta in enumerate([5, 12, 18, 22, 27, 31, 40, 55]):
        recep = now - timedelta(minutes=tdelta + 20)
        diag  = recep + timedelta(minutes=np.random.randint(8, 18))
        com   = diag + timedelta(minutes=np.random.randint(1, 8))
        ejemplos.append({
            "case_code": f"IO-{now.strftime('%Y%m%d')}-{100+i}",
            "historia_clinica": f"HC{3000+i}",
            "paciente": f"P{i+1}",
            "servicio": np.random.choice(["Cirugía General","Ginecología","Cabeza y Cuello","Trauma"]),
            "cirujano": np.random.choice(["Dr. A","Dra. B","Dr. C"]),
            "pieza": np.random.choice(["Ganglio centinela","Borde mamario","Tiroides lóbulo","Colon segmento"]),
            "estado": "reportado",
            "t_inicio": (recep - timedelta(minutes=5)).isoformat(timespec="minutes"),
            "t_recepcion": recep.isoformat(timespec="minutes"),
            "t_criostato": (recep + timedelta(minutes=5)).isoformat(timespec="minutes"),
            "t_diagnostico": diag.isoformat(timespec="minutes"),
            "t_comunicado": com.isoformat(timespec="minutes"),
            "notas": "demo"
        })
    with con:
        for row in ejemplos:
            insert_case(con, **row)

# ============================
# Interfaz
# ============================
st.set_page_config(page_title=APP_TITLE, page_icon="🧊", layout="wide")
st.title(APP_TITLE)
st.caption("Registro de hitos, TAT y cumplimiento de SLA — Zona horaria: America/Lima")

# Sidebar
st.sidebar.header("Configuración")
db_path = st.sidebar.text_input("Ruta de base de datos SQLite", value=DB_PATH_DEFAULT, help="Se crea automáticamente si no existe.")
if st.sidebar.button("Inicializar/Verificar BD", use_container_width=True):
    init_db(db_path)
    refresh_cache()
    st.sidebar.success("Base de datos lista.")

sla_min = st.sidebar.number_input("SLA total (minutos) — recepción→comunicación", min_value=5, max_value=120, value=30, step=5)
f_inicio = st.sidebar.date_input("Filtro desde (fecha recepción)", value=date.today())
f_fin    = st.sidebar.date_input("Filtro hasta (fecha recepción)", value=date.today())

# Conexión inicial
init_db(db_path)
con = get_connection(db_path)

# Pestañas
tab_reg, tab_dia, tab_analit, tab_admin = st.tabs(
    ["➕ Registro / Hitos", "📋 Casos filtrados", "📈 Análisis", "🛠️ Administración"]
)

# ----------------------------
# Registro / Hitos
# ----------------------------
with tab_reg:
    st.subheader("Nuevo caso")
    with st.form("form_new_case", clear_on_submit=True):
        cols = st.columns(3)
        case_code = cols[0].text_input("Código de caso (opcional)", value="")
        hc = cols[1].text_input("Historia clínica", value="")
        paciente = cols[2].text_input("Paciente (iniciales / seudónimo)", value="")
        cols2 = st.columns(3)
        servicio = cols2[0].selectbox("Servicio", ["Cirugía General","Ginecología","Cabeza y Cuello","Trauma","Otro"])
        cirujano = cols2[1].text_input("Cirujano", value="")
        pieza = cols2[2].text_input("Espécimen", value="")
        t_inicio_now = st.checkbox("Usar ahora como t_inicio", value=True)
        t_inicio = datetime.now(TZ).replace(second=0, microsecond=0) if t_inicio_now else datetime.combine(date.today(), time(0,0), TZ)
        notas = st.text_area("Notas", height=80)
        submitted = st.form_submit_button("Crear caso", use_container_width=True)
        if submitted:
            insert_case(con,
                        case_code=case_code or None,
                        historia_clinica=hc or None,
                        paciente=paciente or None,
                        servicio=servicio or None,
                        cirujano=cirujano or None,
                        pieza=pieza or None,
                        estado="pendiente",
                        t_inicio=t_inicio.isoformat(timespec="minutes"),
                        t_recepcion=None, t_criostato=None, t_diagnostico=None, t_comunicado=None,
                        notas=notas or None)
            refresh_cache()
            st.success("Caso creado.")
            st.rerun()  # <- asegura que el resto de la página se regenere con el nuevo caso

    st.divider()
    st.subheader("Actualizar hitos de tiempo")

    # Recarga SIEMPRE antes de usar
    df = load_cases(db_path)

    if df.empty:
        st.info("No hay casos. Registre uno arriba.")
    else:
        df_show = df[["id","case_code","historia_clinica","paciente","servicio","estado"]].copy()
        df_show["rotulo"] = df_show.apply(lambda r: f'#{r["id"]} · {r["case_code"] or ""} · {r["paciente"] or ""} · {r["servicio"] or ""} · {r["estado"]}', axis=1)
        selected = st.selectbox("Seleccione caso", options=df_show["id"].tolist(), format_func=lambda x: df_show.loc[df_show["id"]==x, "rotulo"].values[0])

        col0, colA, colB, colC = st.columns(4)
        if col0.button("Marcar recepción = ahora", use_container_width=True):
            update_case_time(con, selected, "t_recepcion", datetime.now(TZ).replace(second=0, microsecond=0).isoformat(timespec="minutes"))
            refresh_cache(); st.rerun()
        if colA.button("Marcar criostato = ahora", use_container_width=True):
            update_case_time(con, selected, "t_criostato", datetime.now(TZ).replace(second=0, microsecond=0).isoformat(timespec="minutes"))
            refresh_cache(); st.rerun()
        if colB.button("Marcar diagnóstico = ahora", use_container_width=True):
            update_case_time(con, selected, "t_diagnostico", datetime.now(TZ).replace(second=0, microsecond=0).isoformat(timespec="minutes"))
            refresh_cache(); st.rerun()
        if colC.button("Marcar comunicación = ahora", use_container_width=True):
            update_case_time(con, selected, "t_comunicado", datetime.now(TZ).replace(second=0, microsecond=0).isoformat(timespec="minutes"))
            update_case_general(con, selected, estado="reportado")
            refresh_cache(); st.rerun()

        with st.expander("Editar manualmente (corregir hora/minuto)"):
            c1, c2 = st.columns(2)
            dt_recep = c1.time_input("Hora de recepción", value=datetime.now(TZ).time().replace(second=0, microsecond=0))
            if c1.button("Guardar recepción (h:m)"):
                dt = datetime.combine(date.today(), dt_recep, TZ)
                update_case_time(con, selected, "t_recepcion", dt.isoformat(timespec="minutes"))
                refresh_cache(); st.rerun()
            dt_crio = c2.time_input("Hora de criostato", value=datetime.now(TZ).time().replace(second=0, microsecond=0))
            if c2.button("Guardar criostato (h:m)"):
                dt = datetime.combine(date.today(), dt_crio, TZ)
                update_case_time(con, selected, "t_criostato", dt.isoformat(timespec="minutes"))
                refresh_cache(); st.rerun()
            c3, c4 = st.columns(2)
            dt_diag = c3.time_input("Hora de diagnóstico", value=datetime.now(TZ).time().replace(second=0, microsecond=0))
            if c3.button("Guardar diagnóstico (h:m)"):
                dt = datetime.combine(date.today(), dt_diag, TZ)
                update_case_time(con, selected, "t_diagnostico", dt.isoformat(timespec="minutes"))
                refresh_cache(); st.rerun()
            dt_com = c4.time_input("Hora de comunicación", value=datetime.now(TZ).time().replace(second=0, microsecond=0))
            if c4.button("Guardar comunicación (h:m)"):
                dt = datetime.combine(date.today(), dt_com, TZ)
                update_case_time(con, selected, "t_comunicado", dt.isoformat(timespec="minutes"))
                update_case_general(con, selected, estado="reportado")
                refresh_cache(); st.rerun()

# ----------------------------
# Casos filtrados
# ----------------------------
with tab_dia:
    st.subheader("Casos en rango seleccionado")

    df2 = load_cases(db_path)  # recarga
    if df2.empty:
        st.info("No hay casos en la base.")
        show = pd.DataFrame(columns=[
            "id","case_code","historia_clinica","paciente","servicio","cirujano","pieza","estado",
            "t_recepcion","t_criostato","t_diagnostico","t_comunicado",
            "min_recep_a_diag","min_diag_a_com","min_total","cumple_SLA"
        ])
    else:
        df2["t_recepcion_dt"] = pd.to_datetime(df2["t_recepcion"], errors="coerce")
        if df2["t_recepcion_dt"].notna().any():
            start = datetime.combine(f_inicio, time(0,0), TZ)
            end   = datetime.combine(f_fin, time(23,59), TZ)
            mask = df2["t_recepcion_dt"].between(start, end, inclusive="both")
            df2 = df2[mask]
        else:
            df2 = df2.iloc[0:0]

        df2 = compute_metrics(df2, sla_min)

        view_cols = [
            "id","case_code","historia_clinica","paciente","servicio","cirujano","pieza","estado",
            "t_recepcion","t_criostato","t_diagnostico","t_comunicado",
            "min_recep_a_diag","min_diag_a_com","min_total","cumple_SLA"
        ]
        show = df2[view_cols].copy() if not df2.empty else pd.DataFrame(columns=view_cols)

    # Formato seguro (convierte a num antes de redondear)
    for c in ["min_recep_a_diag","min_diag_a_com","min_total"]:
        if c in show.columns:
            show[c] = pd.to_numeric(show[c], errors="coerce").round(1)

    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button(
        "Descargar CSV",
        data=show.to_csv(index=False).encode("utf-8"),
        file_name="intraop_filtrado.csv",
        mime="text/csv"
    )

# ----------------------------
# Análisis
# ----------------------------
with tab_analit:
    st.subheader("KPIs y gráficos")
    df3 = compute_metrics(load_cases(db_path).copy(), sla_min)

    if df3.empty or df3["min_total"].dropna().empty:
        st.info("Sin datos suficientes para KPIs. Registre y complete hitos.")
    else:
        kpis = summarize(df3, sla_min)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Casos totales", kpis["n"])
        c2.metric("Casos con TAT", kpis["n_con_TAT"])
        c3.metric("Mediana TAT (min)", f'{kpis["mediana_min"]:.1f}')
        c4.metric("Cumplimiento SLA (%)", f'{kpis["cumplimiento_%"]:.1f}')
        c5, c6 = st.columns(2)
        c5.metric("P90 TAT (min)", f'{kpis["p90_min"]:.1f}')
        c6.metric("Prom. recepción→diagnóstico (min)", f'{kpis["prom_recep_diag"]:.1f}')
        st.divider()

        plot_df = df3[df3["min_total"].notna()][["id","case_code","servicio","min_total","cumple_SLA"]].copy()
        if not plot_df.empty:
            plot_df["rotulo"] = plot_df.apply(lambda r: f'#{r["id"]} {r["case_code"] or ""} · {r["servicio"] or ""}', axis=1)

            chart = alt.Chart(plot_df).mark_bar().encode(
                x=alt.X("min_total:Q", title="TAT total (min)"),
                y=alt.Y("rotulo:N", sort="-x", title="Caso"),
                color=alt.Color("cumple_SLA:N", legend=alt.Legend(title="Cumple SLA")),
                tooltip=["id","case_code","servicio","min_total","cumple_SLA"]
            ).properties(height=400)
            sla_rule = alt.Chart(pd.DataFrame({"sla":[sla_min]})).mark_rule(strokeDash=[5,5]).encode(x="sla:Q")
            st.altair_chart(chart + sla_rule, use_container_width=True)

            hist = alt.Chart(plot_df).mark_bar().encode(
                x=alt.X("min_total:Q", bin=alt.Bin(maxbins=20), title="TAT total (min)"),
                y=alt.Y("count()", title="Frecuencia")
            ).properties(height=300)
            st.altair_chart(hist, use_container_width=True)
        else:
            st.info("Sin TAT calculado para graficar.")

# ----------------------------
# Administración
# ----------------------------
with tab_admin:
    st.subheader("Utilidades")
    cA, cB, cC = st.columns(3)
    if cA.button("Sembrar datos de ejemplo", use_container_width=True):
        seed_demo(con); refresh_cache(); st.success("Se agregaron casos de ejemplo."); st.rerun()
    if cB.button("Marcar caso más reciente como 'pendiente'", use_container_width=True):
        df = load_cases(db_path)
        if not df.empty:
            update_case_general(con, int(df.iloc[0]["id"]), estado="pendiente")
            refresh_cache(); st.success("Actualizado."); st.rerun()
    if cC.button("Marcar caso más reciente como 'reportado'", use_container_width=True):
        df = load_cases(db_path)
        if not df.empty:
            update_case_general(con, int(df.iloc[0]["id"]), estado="reportado")
            refresh_cache(); st.success("Actualizado."); st.rerun()

    st.caption("Nota: el archivo SQLite intraop.db persiste en el directorio de la app.")
