# app.py
# Tablero Intraoperatorio (Congelación) - Streamlit
# Autor: Sabino (coordinación de Histología) · Asistente: usted
# Finalidad: Registrar y monitorizar TAT intraoperatorio y cumplimiento de SLA.

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

# ----------------------------
# Utilidades de persistencia
# ----------------------------
def init_db(db_path: str):
    with closing(sqlite3.connect(db_path)) as con, con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_code TEXT,          -- código o nro. caso intraop
            historia_clinica TEXT,
            paciente TEXT,           -- iniciales o seudónimo
            servicio TEXT,
            cirujano TEXT,
            pieza TEXT,              -- espécimen
            estado TEXT,             -- pendiente | reportado
            t_inicio TEXT,           -- ingreso a tablero (ISO)
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

# ----------------------------
# Cálculos de métricas
# ----------------------------
def parse_iso(dt_iso):
    if pd.isna(dt_iso) or dt_iso is None or dt_iso == "":
        return None
    try:
        return datetime.fromisoformat(dt_iso)
    except Exception:
        return None

def duration_minutes(t0, t1):
    if t0 is None or t1 is None:
        return np.nan
    return (t1 - t0).total_seconds() / 60.0

def compute_metrics(df: pd.DataFrame, sla_min: int) -> pd.DataFrame:
    # Convertir tiempos
    for col in ["t_recepcion", "t_criostato", "t_diagnostico", "t_comunicado", "t_inicio"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_iso)

    # Derivadas
    df["min_recep_a_diag"]  = df.apply(lambda r: duration_minutes(r["t_recepcion"],  r["t_diagnostico"]), axis=1)
    df["min_diag_a_com"]    = df.apply(lambda r: duration_minutes(r["t_diagnostico"], r["t_comunicado"]), axis=1)
    df["min_total"]         = df.apply(lambda r: duration_minutes(r["t_recepcion"],  r["t_comunicado"]), axis=1)

    # Cumplimiento
    df["cumple_SLA"] = df["min_total"] <= sla_min

    # Estado reportado si tiene t_diagnostico y t_comunicado
    df["estado_calc"] = np.where(df["t_diagnostico"].notna() & df["t_comunicado"].notna(), "reportado", "pendiente")

    return df

def summarize(df: pd.DataFrame, sla_min: int) -> dict:
    df_ok = df[df["min_total"].notna()]
    if df_ok.empty:
        return {"n": 0}
    s = {
        "n": int(len(df)),
        "n_con_TAT": int(len(df_ok)),
        "mediana_min": float(np.nanmedian(df_ok["min_total"])),
        "p90_min": float(np.nanpercentile(df_ok["min_total"], 90)),
        "cumplimiento_%": float(100.0 * (df_ok["min_total"] <= sla_min).mean()),
        "prom_recep_diag": float(np.nanmean(df_ok["min_recep_a_diag"])),
        "prom_diag_com": float(np.nanmean(df_ok["min_diag_a_com"])),
    }
    return s

# ----------------------------
# Datos de ejemplo (opcional)
# ----------------------------
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
            "t_inicio": (recep - timedelta(minutes=5)).astimezone(TZ).isoformat(timespec="minutes"),
            "t_recepcion": recep.astimezone(TZ).isoformat(timespec="minutes"),
            "t_criostato": (recep + timedelta(minutes=5)).astimezone(TZ).isoformat(timespec="minutes"),
            "t_diagnostico": diag.astimezone(TZ).isoformat(timespec="minutes"),
            "t_comunicado": com.astimezone(TZ).isoformat(timespec="minutes"),
            "notas": "demo"
        })
    with con:
        for row in ejemplos:
            insert_case(con, **row)

# ----------------------------
# Interfaz
# ----------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🧊", layout="wide")
st.title(APP_TITLE)
st.caption("Registro de hitos, TAT y cumplimiento de SLA — Zona horaria: America/Lima")

# Sidebar: configuración
st.sidebar.header("Configuración")
db_path = st.sidebar.text_input("Ruta de base de datos SQLite", value=DB_PATH_DEFAULT, help="Se crea automáticamente si no existe.")
if st.sidebar.button("Inicializar/Verificar BD", use_container_width=True):
    init_db(db_path)
    refresh_cache()
    st.sidebar.success("Base de datos lista.")

# SLA y filtros
sla_min = st.sidebar.number_input("SLA total (minutos) — recepción→comunicación", min_value=5, max_value=120, value=30, step=5)
f_inicio = st.sidebar.date_input("Filtro desde (fecha recepción)", value=date.today())
f_fin    = st.sidebar.date_input("Filtro hasta (fecha recepción)", value=date.today())

# Conexión y carga
init_db(db_path)
con = get_connection(db_path)
df = load_cases(db_path)

# Pestañas
tab_reg, tab_dia, tab_analit, tab_admin = st.tabs(["➕ Registro / Hitos", "📋 Casos filtrados", "📈 Análisis", "🛠️ Administración"])

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

    st.divider()
    st.subheader("Actualizar hitos de tiempo")
    if df.empty:
        st.info("No hay casos. Registre uno arriba.")
    else:
        # Selector de caso
        df_show = df[["id","case_code","historia_clinica","paciente","servicio","estado"]].copy()
        df_show["rotulo"] = df_show.apply(lambda r: f'#{r["id"]} · {r["case_code"] or ""} · {r["paciente"] or ""} · {r["servicio"] or ""} · {r["estado"]}', axis=1)
        selected = st.selectbox("Seleccione caso", options=df_show["id"].tolist(), format_func=lambda x: df_show.loc[df_show["id"]==x, "rotulo"].values[0])
        colA, colB, colC, colD = st.columns(4)
        now_btn = st.button("Marcar recepción = ahora", use_container_width=True)
        if now_btn:
            update_case_time(con, selected, "t_recepcion", datetime.now(TZ).replace(second=0, microsecond=0).isoformat(timespec="minutes"))
            refresh_cache()
        if colA.button("Marcar criostato = ahora", use_container_width=True):
            update_case_time(con, selected, "t_criostato", datetime.now(TZ).replace(second=0, microsecond=0).isoformat(timespec="minutes"))
            refresh_cache()
        if colB.button("Marcar diagnóstico = ahora", use_container_width=True):
            update_case_time(con, selected, "t_diagnostico", datetime.now(TZ).replace(second=0, microsecond=0).isoformat(timespec="minutes"))
            refresh_cache()
        if colC.button("Marcar comunicación = ahora", use_container_width=True):
            update_case_time(con, selected, "t_comunicado", datetime.now(TZ).replace(second=0, microsecond=0).isoformat(timespec="minutes"))
            update_case_general(con, selected, estado="reportado")
            refresh_cache()

        with st.expander("Editar manualmente (corregir hora/minuto)"):
            c1, c2 = st.columns(2)
            dt_recep = c1.time_input("Hora de recepción", value=datetime.now(TZ).time().replace(second=0, microsecond=0))
            if c1.button("Guardar recepción (h:m)"):
                dt = datetime.combine(date.today(), dt_recep, TZ)
                update_case_time(con, selected, "t_recepcion", dt.isoformat(timespec="minutes"))
                refresh_cache()
            dt_crio = c2.time_input("Hora de criostato", value=datetime.now(TZ).time().replace(second=0, microsecond=0))
            if c2.button("Guardar criostato (h:m)"):
                dt = datetime.combine(date.today(), dt_crio, TZ)
                update_case_time(con, selected, "t_criostato", dt.isoformat(timespec="minutes"))
                refresh_cache()
            c3, c4 = st.columns(2)
            dt_diag = c3.time_input("Hora de diagnóstico", value=datetime.now(TZ).time().replace(second=0, microsecond=0))
            if c3.button("Guardar diagnóstico (h:m)"):
                dt = datetime.combine(date.today(), dt_diag, TZ)
                update_case_time(con, selected, "t_diagnostico", dt.isoformat(timespec="minutes"))
                refresh_cache()
            dt_com = c4.time_input("Hora de comunicación", value=datetime.now(TZ).time().replace(second=0, microsecond=0))
            if c4.button("Guardar comunicación (h:m)"):
                dt = datetime.combine(date.today(), dt_com, TZ)
                update_case_time(con, selected, "t_comunicado", dt.isoformat(timespec="minutes"))
                update_case_general(con, selected, estado="reportado")
                refresh_cache()

# ----------------------------
# Casos filtrados (tabla del día)
# ----------------------------
with tab_dia:
    st.subheader("Casos en rango seleccionado")
    df2 = df.copy()
    # Parse recepcion para filtrar por fecha
    df2["t_recepcion_dt"] = df2["t_recepcion"].apply(parse_iso)
    mask = True
    if not df2["t_recepcion_dt"].isna().all():
        mask = df2["t_recepcion_dt"].between(
            datetime.combine(f_inicio, time(0,0), TZ),
            datetime.combine(f_fin, time(23,59), TZ)
        )
        df2 = df2[mask.fillna(False)]
    df2 = compute_metrics(df2, sla_min)
    # Selección de columnas legibles
    view_cols = [
        "id","case_code","historia_clinica","paciente","servicio","cirujano","pieza","estado",
        "t_recepcion","t_criostato","t_diagnostico","t_comunicado",
        "min_recep_a_diag","min_diag_a_com","min_total","cumple_SLA"
    ]
    show = df2[view_cols].copy() if not df2.empty else pd.DataFrame(columns=view_cols)
    # Formatos
    num_cols = ["min_recep_a_diag","min_diag_a_com","min_total"]
    for c in num_cols:
        if c in show.columns:
            show[c] = show[c].round(1)
    st.dataframe(show, use_container_width=True, hide_index=True)

    # Descarga CSV
    csv = show.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar CSV", data=csv, file_name="intraop_filtrado.csv", mime="text/csv")

# ----------------------------
# Análisis
# ----------------------------
with tab_analit:
    st.subheader("KPIs y gráficos")
    df3 = compute_metrics(df.copy(), sla_min)
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

        # Barras por caso
        plot_df = df3[df3["min_total"].notna()][["id","case_code","servicio","min_total","cumple_SLA"]].copy()
        plot_df["rotulo"] = plot_df.apply(lambda r: f'#{r["id"]} {r["case_code"] or ""} · {r["servicio"] or ""}', axis=1)
        chart = alt.Chart(plot_df).mark_bar().encode(
            x=alt.X("min_total:Q", title="TAT total (min)"),
            y=alt.Y("rotulo:N", sort="-x", title="Caso"),
            color=alt.Color("cumple_SLA:N", legend=alt.Legend(title="Cumple SLA")),
            tooltip=["id","case_code","servicio","min_total","cumple_SLA"]
        ).properties(height=400)
        sla_rule = alt.Chart(pd.DataFrame({"sla":[sla_min]})).mark_rule(strokeDash=[5,5]).encode(x="sla:Q")
        st.altair_chart(chart + sla_rule, use_container_width=True)

        # Distribución
        hist = alt.Chart(plot_df).mark_bar().encode(
            x=alt.X("min_total:Q", bin=alt.Bin(maxbins=20), title="TAT total (min)"),
            y=alt.Y("count()", title="Frecuencia")
        ).properties(height=300)
        st.altair_chart(hist, use_container_width=True)

# ----------------------------
# Administración
# ----------------------------
with tab_admin:
    st.subheader("Utilidades")
    cA, cB, cC = st.columns(3)
    if cA.button("Sembrar datos de ejemplo", use_container_width=True):
        seed_demo(con)
        refresh_cache()
        st.success("Se agregaron casos de ejemplo.")
    if cB.button("Marcar caso seleccionado como 'pendiente'", use_container_width=True):
        if not df.empty:
            update_case_general(con, int(df.iloc[0]["id"]), estado="pendiente")
            refresh_cache()
            st.success("Caso marcado como pendiente (el más reciente).")
    if cC.button("Marcar caso seleccionado como 'reportado'", use_container_width=True):
        if not df.empty:
            update_case_general(con, int(df.iloc[0]["id"]), estado="reportado")
            refresh_cache()
            st.success("Caso marcado como reportado (el más reciente).")

    st.caption("Nota: el archivo SQLite intraop.db persiste en el directorio de la app.")
