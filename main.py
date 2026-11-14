"""Monitor de TIDIS - Streamlit page.

Every visit (at most once per hour) downloads the file of outstanding securities,
saves the TIDIS "image" of the day, rebuilds the daily history (imputing missing
days), makes/saves the forecasts of the day and flags significant changes.
Run the same cycle without the page with:  python collect.py
"""
from __future__ import annotations

import dataclasses
import hashlib
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


def code_version() -> str:
    """Fingerprint of the tidis package source."""
    files = sorted((Path(__file__).resolve().parent / "tidis").glob("*.py"))
    return hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()[:16]


# Streamlit Cloud keeps the Python process alive when new code is pulled: it re-runs this script,
# but modules already imported (tidis.*) and st.cache_data results stay in memory. If they come
# from another version of the code, throw them away so the new code is really used.
CODE_VERSION = code_version()
if "tidis" in sys.modules and getattr(sys.modules["tidis"], "__code_version__", None) != CODE_VERSION:
    for _name in [m for m in sys.modules if m == "tidis" or m.startswith("tidis.")]:
        del sys.modules[_name]
    st.cache_data.clear()

import tidis  # noqa: E402
from tidis import pipeline  # noqa: E402
from tidis.alerts import METRIC_LABELS, SEV_ORDER, forecast_accuracy  # noqa: E402
from tidis.analysis import compare_images  # noqa: E402
from tidis.config import SETTINGS, SOURCE_URL  # noqa: E402
from tidis.forecast import METHODS, calibrate_alpha, run_forecasts, with_options  # noqa: E402
from tidis.storage import Store  # noqa: E402

tidis.__code_version__ = CODE_VERSION

st.set_page_config(page_title="Monitor TIDIS", page_icon="📉", layout="wide")

UNIT = 1e9
UNIT_LABEL = "miles de millones COP"
METHOD_COLORS = alt.Scale(domain=list(METHODS.values()), range=["#f58518", "#4c78a8"])
COMPONENT_COLORS = alt.Scale(domain=["Emitido", "Usado (redimido)", "Vencido sin usar"],
                             range=["#54a24b", "#4c78a8", "#e45756"])
store = Store()


def cop(x: float) -> str:
    return f"{'-' if x < 0 else ''}${abs(x):,.0f}"


def mm(x: float) -> str:
    """Thousands of millions of COP; sign first so st.metric colours deltas correctly."""
    return f"{'-' if x < 0 else ''}${abs(x) / UNIT:,.1f} mil M"


def pct(x: float, digits: int = 1) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{digits}f}%"


# --------------------------------------------------------------------------- data
@st.cache_data(ttl=3600, show_spinner="Descargando el archivo de títulos vigentes…")
def auto_fetch(hour_key: str) -> dict:
    """Automatic download, at most once per hour per server process."""
    return pipeline.fetch_and_ingest(store)


@st.cache_data(show_spinner="Reconstruyendo la historia de los TIDIS…")
def load(version: float, code: str = CODE_VERSION) -> pipeline.History:
    h = pipeline.load_history(store)
    if not h.empty:
        h.alerts = pipeline.refresh_alerts(store, h)
    return h


@st.cache_data(show_spinner="Calculando índices y predicciones…", max_entries=64)
def live_forecast(version: float, asof: pd.Timestamp, alpha: float, new_issues: bool, code: str = CODE_VERSION):
    h = load(version, code)
    return run_forecasts(h.panel, h.alive_on_maturity, h.snapshots,
                         with_options(SETTINGS, include_new_issuance=new_issues), alpha=alpha, asof=asof)


@st.cache_data(show_spinner="Probando valores de α con la historia (backtest)…", max_entries=8)
def calibration(version: float, code: str = CODE_VERSION) -> pd.DataFrame:
    h = load(version, code)
    return calibrate_alpha(h.panel, h.alive_on_maturity, h.snapshots, h.actual_obs)


def data_version() -> float:
    # alerts.csv is excluded on purpose: it is an output of load()
    return sum(p.stat().st_mtime_ns for p in (store.catalog_path, store.vintages_path) if p.exists())


def show_fetch_result(res: dict | None, where=st.sidebar) -> None:
    if not res:
        return
    if res["status"] == "error":
        where.warning(res["error"] + "\n\nSe muestra la historia guardada.")
    elif res["status"] == "unchanged":
        where.info(f"Imagen del {res['date']} ya estaba guardada (sin cambios).")
    else:
        msg = "nueva" if res["status"] == "new" else "actualizada"
        extra = f" · {res['imputed_rows']} filas imputadas" if res.get("imputed_rows") else ""
        where.success(f"Imagen del {res['date']} {msg}: {res['rows']} títulos, {cop(res['total'])}{extra}")


def alert_rules(alerts: pd.DataFrame, categories=None):
    a = alerts[alerts["severity"] == "alerta"]
    if categories:
        a = a[a["category"].isin(categories)]
    a = a.drop_duplicates("date")
    if a.empty:
        return []
    return [alt.Chart(a).mark_rule(color="#e45756", strokeDash=[3, 3]).encode(
        x="date:T", tooltip=[alt.Tooltip("date:T", title="alerta"), alt.Tooltip("message:N", title="detalle")])]


# --------------------------------------------------------------------------- sidebar: data
st.sidebar.title("📉 Monitor TIDIS")
if st.sidebar.button("Descargar ahora", type="primary", width="stretch"):
    auto_fetch.clear()
    with st.spinner("Descargando…"):
        st.session_state["manual_fetch"] = pipeline.fetch_and_ingest(store)
fetch_result = st.session_state.pop("manual_fetch", None) or auto_fetch(pipeline.now_local().strftime("%Y-%m-%d %H"))
show_fetch_result(fetch_result)

with st.sidebar.expander("Agregar archivo histórico"):
    st.caption("Sube un archivo 'características financieras títulos vigentes' de una fecha pasada "
               "para completar la historia.")
    up = st.file_uploader("Archivo .xlsx", type=["xlsx"])
    up_date = st.date_input("Fecha de la imagen (vacío = leerla del archivo)", value=None,
                            max_value=pipeline.today())
    if up is not None and st.button("Guardar imagen"):
        try:
            show_fetch_result(pipeline.ingest(store, up.getvalue(), day=up_date, origin="carga manual"))
        except Exception as exc:
            st.error(f"No se pudo leer el archivo: {exc}")

n_new = pipeline.refresh_forecasts(store)   # forecast vintage for every new image (write-once)
version = data_version()
h = load(version, CODE_VERSION)
if any(not hasattr(h, f.name) for f in dataclasses.fields(pipeline.History)):
    # object cached by an incompatible version of the code: drop every cached result and rebuild
    st.cache_data.clear()
    h = load(version, CODE_VERSION)
    if any(not hasattr(h, f.name) for f in dataclasses.fields(pipeline.History)):
        st.error("La aplicación cargó una versión antigua del código. Reiníciela (Manage app → Reboot app).")
        st.stop()

st.title("TIDIS: evolución diaria, depreciación, predicciones y alertas")
if h.empty:
    st.info("Todavía no hay imágenes guardadas. Descarga el archivo (barra lateral) o sube uno histórico.")
    st.stop()

# --------------------------------------------------------------------------- sidebar: forecast controls
obs_days = list(h.obs.index)
st.sidebar.divider()
st.sidebar.subheader("Índice y predicción")
base_day = st.sidebar.selectbox("Imagen base", obs_days[::-1], format_func=lambda d: f"{d.date()}",
                                help="Las predicciones usan solo información disponible hasta esta imagen")
alpha = st.sidebar.slider("Suavizado α del índice de depreciación", 0.01, 1.0, float(SETTINGS.alpha), 0.01,
                          help="Peso del dato más reciente. α pequeño = tendencia más suave (memoria larga); "
                               "α = 1 = sin suavizar.")
new_issues = st.sidebar.checkbox("Incluir nuevas emisiones", value=SETTINGS.include_new_issuance)
if abs(alpha - SETTINGS.alpha) > 1e-9:
    st.sidebar.caption(f"Las predicciones guardadas y las alertas usan α = {SETTINGS.alpha:g} (config).")
fc = live_forecast(version, base_day, alpha, new_issues, CODE_VERSION)

st.sidebar.divider()
st.sidebar.caption(f"Fuente: [archivo de títulos vigentes]({SOURCE_URL})  \n"
                   f"Historia guardada en `{store.root}`  \n"
                   f"{len(h.catalog)} imágenes observadas · {int(h.imputed.sum())} días imputados")

# --------------------------------------------------------------------------- header
last_day = h.panel.index[-1]
image = h.panel.loc[last_day]
age = (pd.Timestamp(pipeline.today()) - last_day).days
if age > SETTINGS.stale_days:
    st.warning(f"La última imagen es del {last_day.date()} ({age} días). Las cifras de hoy no están observadas; "
               "ver la pestaña *Predicciones* para la estimación.")

prev_obs = obs_days[-2] if len(obs_days) > 1 else None
total = float(image.sum())
last30 = h.flows[h.flows.index > last_day - pd.Timedelta(days=30)]
latest_v = h.vintages.iloc[-1] if not h.vintages.empty else None
ser = h.index.series

c = st.columns(5)
prev_total = float(h.obs.loc[prev_obs].sum()) if prev_obs is not None else np.nan
c[0].metric("Saldo TIDIS en circulación", mm(total),
            delta=f"{pct(100 * (total / prev_total - 1), 2)} ({mm(total - prev_total)})" if prev_obs is not None else None,
            help=f"{cop(total)} al {last_day.date()}" + (f" · cambio vs {prev_obs.date()}" if prev_obs is not None else ""))
c[1].metric("Usado (redimido) 30 días", mm(last30["redeemed"].sum()),
            help=f"{100 * last30['redeemed'].sum() / total:.1f}% del saldo actual")
c[2].metric("Emitido 30 días", mm(last30["issued"].sum()),
            help=f"{100 * last30['issued'].sum() / total:.1f}% del saldo actual")
if h.index.kind == "index" and len(ser) > 7:
    lvl, lvl7 = ser["rate_smooth"].iloc[-1], ser["rate_smooth"].iloc[-8]
    c[3].metric(f"Depreciación diaria (α={SETTINGS.alpha:g})", f"{100 * lvl:.3f}%/día",
                delta=f"{pct(100 * (lvl / lvl7 - 1), 0)} vs 7 días", delta_color="off",
                help="Fracción del saldo vigente que se usa por día, suavizada (índice de varias imágenes)")
if latest_v is not None:
    unused = latest_v.get("multi_expected_unused", latest_v.get("expected_unused_at_maturity", np.nan))
    c[4].metric("Vencerá sin usarse (pred.)", mm(unused), help=f"{100 * unused / total:.1f}% del saldo actual · "
                "predicción guardada con varias imágenes")

recent = h.alerts[(h.alerts["date"] >= last_day - pd.Timedelta(days=6)) & (h.alerts["severity"] != "info")]
recent = recent.assign(o=recent["severity"].map(SEV_ORDER)).sort_values(["o", "date"], ascending=[True, False])
for _, a in recent.head(4).iterrows():
    (st.error if a["severity"] == "alerta" else st.warning)(
        f"**{a['date'].date()} · {METRIC_LABELS.get(a['metric'], a['metric'])}** — {a['message']}")
if len(recent) > 4:
    st.caption(f"… y {len(recent) - 4} alertas más en la pestaña *Alertas*.")

tabs = st.tabs(["Imagen actual", "Historia", "Comparar imágenes", "Índice de depreciación", "Predicciones",
                "Alertas", "Datos"])


def image_vs_reference(img: pd.Series, ref: pd.Series | None, ref_day, img_label: str, key: str) -> None:
    """Daily chart of an image (bars by maturity date) with a reference image and the change."""
    df = pd.DataFrame({"saldo": img[img > 0] / UNIT})
    df.index.name = "vencimiento"
    layers = [alt.Chart(df.reset_index()).mark_bar(color="#9ecae9").encode(
        x=alt.X("vencimiento:T", title="Fecha de vencimiento"), y=alt.Y("saldo:Q", title=f"Saldo ({UNIT_LABEL})"),
        tooltip=[alt.Tooltip("vencimiento:T", title="vencimiento"), alt.Tooltip("saldo:Q", format=",.2f", title=img_label)])]
    if ref is not None:
        r = pd.DataFrame({"saldo": ref[ref > 0] / UNIT})
        r.index.name = "vencimiento"
        layers.append(alt.Chart(r.reset_index()).mark_tick(color="#08519c", thickness=2).encode(
            x="vencimiento:T", y="saldo:Q",
            tooltip=[alt.Tooltip("vencimiento:T", title="vencimiento"),
                     alt.Tooltip("saldo:Q", format=",.2f", title=f"{ref_day.date()}")]))
    st.altair_chart(alt.layer(*layers), width="stretch", key=f"{key}_img")
    if ref is not None:
        st.caption(f"Barras: {img_label} · marcas azules: imagen del {ref_day.date()}")


def change_chart(cmp_df: pd.DataFrame, key: str) -> None:
    ch = cmp_df[["issued", "redeemed", "matured"]].copy() / UNIT
    ch["redeemed"] *= -1
    ch["matured"] *= -1
    ch.columns = ["Emitido", "Usado (redimido)", "Vencido sin usar"]
    ch = ch.reset_index().melt("maturity_date", var_name="componente", value_name="monto")
    ch = ch[ch["monto"] != 0]
    if ch.empty:
        st.caption("Sin cambios entre las dos imágenes.")
        return
    st.altair_chart(alt.Chart(ch).mark_bar().encode(
        x=alt.X("maturity_date:T", title="Fecha de vencimiento"), y=alt.Y("monto:Q", title=f"Cambio ({UNIT_LABEL})"),
        color=alt.Color("componente:N", title=None, scale=COMPONENT_COLORS),
        tooltip=[alt.Tooltip("maturity_date:T", title="vencimiento"), "componente",
                 alt.Tooltip("monto:Q", format=",.3f")]), width="stretch", key=key)


def daily_table(cmp_df: pd.DataFrame, a_label: str, b_label: str) -> None:
    view = cmp_df.copy()
    view.index = view.index.date
    view.index.name = "Fecha de vencimiento"
    view.columns = [a_label, b_label, "Emitido", "Usado", "Vencido sin usar", "Cambio", "Cambio %"]
    view[view.columns[:-1]] = view[view.columns[:-1]] / UNIT
    st.caption(f"Cifras en {UNIT_LABEL}, una fila por fecha de vencimiento")
    num = st.column_config.NumberColumn(format="%.3f")
    st.dataframe(view, column_config={c: num for c in view.columns[:-1]} |
                 {"Cambio %": st.column_config.NumberColumn(format="%.2f%%")})


# --------------------------------------------------------------------------- tab: current image
with tabs[0]:
    st.subheader(f"Saldo por fecha de vencimiento al {last_day.date()}")
    ref_opt = {"Imagen anterior": 1, "Hace 7 días": 7, "Hace 30 días": 30, "Hace 90 días": 90}
    ref_label = st.segmented_control("Comparar con", list(ref_opt), default="Imagen anterior")
    ref_day = None
    if ref_label == "Imagen anterior":
        ref_day = prev_obs
    elif ref_label:
        cand = last_day - pd.Timedelta(days=ref_opt[ref_label])
        ref_day = cand if cand >= h.panel.index[0] else None
    ref = h.panel.loc[ref_day] if ref_day is not None else None
    image_vs_reference(image, ref, ref_day, f"imagen del {last_day.date()}", "cur")
    if ref_day is not None:
        if h.imputed.get(ref_day, False):
            st.caption(f"La imagen del {ref_day.date()} es imputada.")
        cmp_df = compare_images(ref, image, last_day, h.alive_on_maturity)
        st.markdown(f"**Cambio por fecha de vencimiento** ({ref_day.date()} → {last_day.date()}): "
                    f"{pct(100 * (image.sum() / ref.sum() - 1), 2)} en el saldo total")
        change_chart(cmp_df, "cur_change")
        with st.expander("Tabla diaria"):
            daily_table(cmp_df, f"Saldo {ref_day.date()}", f"Saldo {last_day.date()}")

# --------------------------------------------------------------------------- tab: history
with tabs[1]:
    st.subheader("Saldo total en circulación (diario)")
    fl = h.flows.reset_index()
    fl["tipo"] = fl["imputed"].map({True: "imputado", False: "observado"})
    fl["saldo"] = fl["total"] / UNIT
    fl["cambio_pct"] = 100 * fl["total"].pct_change()
    line = alt.Chart(fl).mark_line(color="#4c78a8").encode(
        x=alt.X("date:T", title=None), y=alt.Y("saldo:Q", title=f"Saldo ({UNIT_LABEL})", scale=alt.Scale(zero=False)))
    pts = alt.Chart(fl).mark_circle(size=24).encode(
        x="date:T", y="saldo:Q",
        color=alt.Color("tipo:N", title=None, scale=alt.Scale(domain=["observado", "imputado"], range=["#4c78a8", "#f58518"])),
        tooltip=[alt.Tooltip("date:T", title="fecha"), "tipo", alt.Tooltip("saldo:Q", format=",.2f"),
                 alt.Tooltip("cambio_pct:Q", format="+.2f", title="cambio % diario")])
    st.altair_chart(alt.layer(line, pts, *alert_rules(h.alerts)), width="stretch")
    st.caption("Líneas rojas: días con alertas. Puntos naranja: días sin imagen, imputados.")

    st.subheader("¿Por qué cambia el saldo? (por día)")
    if len(h.flows) < 2:
        st.info("Se necesitan al menos dos imágenes para descomponer los cambios.")
    else:
        comp = h.flows[["issued", "redeemed", "matured"]].iloc[1:] / UNIT
        comp["redeemed"] *= -1
        comp["matured"] *= -1
        comp.columns = list(COMPONENT_COLORS.domain)
        cl = comp.reset_index().melt("date", var_name="componente", value_name="monto")
        st.altair_chart(alt.Chart(cl).mark_bar().encode(
            x=alt.X("date:T", title=None), y=alt.Y("monto:Q", title=f"Cambio diario ({UNIT_LABEL})"),
            color=alt.Color("componente:N", title=None, scale=COMPONENT_COLORS),
            tooltip=[alt.Tooltip("date:T", title="día"), "componente", alt.Tooltip("monto:Q", format=",.2f")]),
            width="stretch")

    st.subheader("Evolución de la imagen (saldo por fecha de vencimiento, día a día)")
    win = st.segmented_control("Imágenes", ["30 días", "90 días", "180 días", "Todas"], default="90 días", key="hm")
    ndays = {"30 días": 30, "90 días": 90, "180 días": 180}.get(win or "90 días")
    pan = h.panel if ndays is None else h.panel[h.panel.index > last_day - pd.Timedelta(days=ndays)]
    hm = (pan / UNIT).rename_axis(index="imagen", columns="vencimiento").stack().rename("saldo").reset_index()
    hm = hm[hm["saldo"] > 0]
    hm["imagen_fin"] = hm["imagen"] + pd.Timedelta(days=1)          # one cell = one day x one day
    hm["vencimiento_fin"] = hm["vencimiento"] + pd.Timedelta(days=1)
    st.altair_chart(alt.Chart(hm).mark_rect().encode(
        x=alt.X("imagen:T", title="Fecha de la imagen"), x2="imagen_fin:T",
        y=alt.Y("vencimiento:T", title="Fecha de vencimiento"), y2="vencimiento_fin:T",
        color=alt.Color("saldo:Q", title=UNIT_LABEL, scale=alt.Scale(scheme="blues")),
        tooltip=[alt.Tooltip("imagen:T", title="imagen"), alt.Tooltip("vencimiento:T", title="vencimiento"),
                 alt.Tooltip("saldo:Q", format=",.3f")]).properties(height=450), width="stretch")

    st.subheader("Degradación por fecha de vencimiento")
    st.caption("% del saldo máximo observado de cada fecha de vencimiento que sigue en circulación "
               "(desde que alcanzó ese máximo), según los días que le faltan para vencer.")
    nobs = (h.panel > 0).sum()
    cand = list(nobs[nobs >= 20].index)
    if cand:
        default = [cand[int(i)] for i in np.unique(np.linspace(0, len(cand) - 1, min(8, len(cand))).round())]
        sel = st.multiselect("Fechas de vencimiento", cand, default=default, format_func=lambda d: f"{d.date()}")
        rows = []
        for m in sel:
            s = h.panel[m]
            s = s[s > 0]
            s = s.loc[s.idxmax():]
            rows.append(pd.DataFrame({"dias_al_vencimiento": (m - s.index).days, "pct": 100 * s.values / s.max(),
                                      "vencimiento": str(m.date())}))
        if rows:
            st.altair_chart(alt.Chart(pd.concat(rows)).mark_line().encode(
                x=alt.X("dias_al_vencimiento:Q", title="Días al vencimiento", scale=alt.Scale(reverse=True)),
                y=alt.Y("pct:Q", title="% del saldo máximo"), color=alt.Color("vencimiento:N", title="Vence el"),
                tooltip=["vencimiento", "dias_al_vencimiento", alt.Tooltip("pct:Q", format=".1f")]), width="stretch")

# --------------------------------------------------------------------------- tab: compare
with tabs[2]:
    st.subheader("Comparar dos imágenes (por fecha de vencimiento)")
    first, last = h.panel.index[0].date(), last_day.date()
    cc = st.columns(2)
    d_a = cc[0].date_input("Imagen A", value=(prev_obs or last_day).date(), min_value=first, max_value=last)
    d_b = cc[1].date_input("Imagen B", value=last, min_value=first, max_value=last)
    d_a, d_b = pd.Timestamp(d_a), pd.Timestamp(d_b)
    for lbl, d in (("A", d_a), ("B", d_b)):
        if h.imputed.get(d, False):
            st.caption(f"La imagen {lbl} ({d.date()}) no fue observada: es imputada.")
    if d_a >= d_b:
        st.info("Elige una imagen A anterior a la imagen B.")
    else:
        ia, ib = h.panel.loc[d_a], h.panel.loc[d_b]
        cmp_df = compare_images(ia, ib, d_b, h.alive_on_maturity)
        tot = cmp_df[["balance_a", "balance_b", "issued", "redeemed", "matured"]].sum()
        base = tot["balance_a"]
        k = st.columns(5)
        k[0].metric("Saldo A", mm(base))
        k[1].metric("Saldo B", mm(tot["balance_b"]), delta=f"{pct(100 * (tot['balance_b'] / base - 1), 2)}")
        k[2].metric("Emitido", mm(tot["issued"]), help=f"{100 * tot['issued'] / base:.2f}% del saldo A")
        k[3].metric("Usado", mm(tot["redeemed"]), help=f"{100 * tot['redeemed'] / base:.2f}% del saldo A")
        k[4].metric("Vencido sin usar", mm(tot["matured"]), help=f"{100 * tot['matured'] / base:.2f}% del saldo A")
        image_vs_reference(ib, ia, d_a, f"imagen B ({d_b.date()})", "cmp")
        change_chart(cmp_df, "cmp_change")
        daily_table(cmp_df, f"Saldo A {d_a.date()}", f"Saldo B {d_b.date()}")

# --------------------------------------------------------------------------- tab: depreciation index
with tabs[3]:
    st.subheader(f"Índice de depreciación (α = {alpha:g}, imagen base {base_day.date()})")
    st.caption("Mide qué parte del saldo de TIDIS se va usando (redimiendo) con el tiempo. "
               "α controla cuánto se suaviza la tendencia: α pequeño = más suave; α = 1 = sin suavizar.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Una imagen** — saldo según los días desde la emisión")
        sg = fc.single
        if sg.kind != "index":
            st.info("La imagen tiene muy pocos vencimientos para construir el índice.")
        else:
            p = sg.profile
            raw = p[p["balance"] > 0]
            ch = alt.layer(
                alt.Chart(raw).mark_circle(size=14, color="#bbbbbb").encode(
                    x=alt.X("age:Q", title="Días desde la emisión"), y=alt.Y("index_raw:Q", title="Índice (100 = emisión)"),
                    tooltip=[alt.Tooltip("age:Q", title="días"), alt.Tooltip("index_raw:Q", format=".1f", title="sin suavizar")]),
                alt.Chart(p).mark_line(color="#f58518").encode(x="age:Q", y="index:Q"),
                alt.Chart(p).mark_line(color="#a0522d", strokeDash=[5, 3]).encode(
                    x="age:Q", y="index_mono:Q",
                    tooltip=[alt.Tooltip("age:Q", title="días"), alt.Tooltip("index_mono:Q", format=".1f", title="índice"),
                             alt.Tooltip("daily_rate:Q", format=".3%", title="uso diario")]),
            ).properties(height=300)
            st.altair_chart(ch, width="stretch")
            st.caption("Gris: saldo de cada día de emisión · naranja: suavizado con α · punteado: índice usado "
                       "para predecir (no creciente)")
            m = st.columns(2)
            m[0].metric("Uso diario equivalente", f"{100 * sg.daily_rate:.3f}%/día")
            m[1].metric("Llega al vencimiento", f"{p['index_mono'].iloc[-1]:.1f}%",
                        help="Parte de un título que, según el índice, vence sin usarse")
    with right:
        st.markdown("**Varias imágenes** — uso diario medido entre imágenes consecutivas")
        mg = fc.multi
        if mg.kind != "index":
            st.info("Se necesitan al menos dos imágenes.")
        else:
            sr = mg.series.reset_index(names="date")
            ch = alt.layer(
                alt.Chart(sr).mark_line(color="#bbbbbb").encode(
                    x=alt.X("date:T", title=None), y=alt.Y("index:Q", title="Índice (100 = primera imagen)", scale=alt.Scale(type="log")),
                    tooltip=[alt.Tooltip("date:T", title="día"), alt.Tooltip("index:Q", format=".2f", title="sin suavizar")]),
                alt.Chart(sr).mark_line(color="#4c78a8").encode(
                    x="date:T", y="index_smooth:Q",
                    tooltip=[alt.Tooltip("date:T", title="día"), alt.Tooltip("index_smooth:Q", format=".2f", title="suavizado")]),
            ).properties(height=300)
            st.altair_chart(ch, width="stretch")
            cap = float(np.nanquantile(sr["rate"], 0.98) * 1.5) if sr["rate"].notna().any() else None
            rate_ch = alt.layer(
                alt.Chart(sr).mark_bar(color="#cccccc", clip=True).encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("rate:Q", title="Uso diario (% del saldo)", axis=alt.Axis(format="%"),
                            scale=alt.Scale(domain=[0, cap], clamp=True) if cap else alt.Undefined),
                    tooltip=[alt.Tooltip("date:T", title="día"), alt.Tooltip("rate:Q", format=".3%", title="sin suavizar")]),
                alt.Chart(sr).mark_line(color="#4c78a8").encode(
                    x="date:T", y="rate_smooth:Q",
                    tooltip=[alt.Tooltip("date:T", title="día"), alt.Tooltip("rate_smooth:Q", format=".3%", title="suavizado")]),
                *alert_rules(h.alerts[h.alerts["date"] <= base_day], ["índice", "flujos"]),
            ).properties(height=200)
            st.altair_chart(rate_ch, width="stretch")
            if cap and (sr["rate"] > cap).any():
                st.caption(f"Eje recortado en {100 * cap:.2f}%: {int((sr['rate'] > cap).sum())} día(s) con uso mayor "
                           f"(máximo {100 * sr['rate'].max():.1f}%).")
            m = st.columns(2)
            raw_last = sr["rate"].iloc[-1]
            m[0].metric("Uso diario suavizado", f"{100 * mg.level:.3f}%/día",
                        delta=f"{pct(100 * (raw_last / mg.level - 1), 0)} último día sin suavizar" if mg.level else None,
                        delta_color="off")
            m[1].metric("Índice suavizado", f"{sr['index_smooth'].iloc[-1]:.1f}",
                        help=f"Base 100 el {sr['date'].iloc[0].date()}")

    if sg.kind == "index" or mg.kind == "index":
        st.markdown("**Uso diario según los días al vencimiento** — los dos índices en la misma escala")
        dtm = np.arange(0, sg.tenor + 1)
        curves = pd.DataFrame({"dias": dtm, METHODS["single"]: sg.rate_for(dtm), METHODS["multi"]: mg.rate_for(dtm)})
        curves = curves.melt("dias", var_name="método", value_name="tasa")
        st.altair_chart(alt.Chart(curves).mark_line().encode(
            x=alt.X("dias:Q", title="Días al vencimiento", scale=alt.Scale(reverse=True)),
            y=alt.Y("tasa:Q", title="Uso diario", axis=alt.Axis(format="%")),
            color=alt.Color("método:N", scale=METHOD_COLORS),
            tooltip=["método", "dias", alt.Tooltip("tasa:Q", format=".3%")]).properties(height=250), width="stretch")

    with st.expander("Calibrar α con la historia (backtest)"):
        st.caption("Para varias imágenes pasadas se predice con cada α (solo con datos hasta esa imagen) y se mide el "
                   "error medio absoluto contra las imágenes observadas en los 60 días siguientes.")
        if len(obs_days) < 20:
            st.info("Se necesitan al menos 20 imágenes para calibrar.")
        elif st.button("Calibrar α"):
            st.session_state["calib"] = True
        if st.session_state.get("calib") and len(obs_days) >= 20:
            cal = calibration(version, CODE_VERSION)
            cal["método"] = cal["method"].map(METHODS)
            st.altair_chart(alt.Chart(cal).mark_line(point=True).encode(
                x=alt.X("alpha:Q", title="α", scale=alt.Scale(type="log")), y=alt.Y("mape:Q", title="Error medio absoluto (%)"),
                color=alt.Color("método:N", scale=METHOD_COLORS),
                tooltip=["método", "alpha", alt.Tooltip("mape:Q", format=".2f"), "n"]), width="stretch")
            best = cal.loc[cal.groupby("method")["mape"].idxmin()]
            st.markdown(" · ".join(f"Mejor α para **{METHODS[r['method']]}**: {r['alpha']:g} "
                                   f"(error {r['mape']:.2f}%)" for _, r in best.iterrows()))

# --------------------------------------------------------------------------- tab: forecasts
with tabs[4]:
    st.subheader(f"Predicción del saldo total desde la imagen del {base_day.date()} (α = {alpha:g})")
    horizon = st.slider("Horizonte (días)", 30, SETTINGS.horizon_days, 180, step=30)
    tot = fc.totals[fc.totals["target_date"] <= base_day + pd.Timedelta(days=horizon)].copy()
    tot["método"] = tot["method"].map(METHODS)
    tot["v"] = tot["total"] / UNIT
    hist = h.flows.reset_index()[["date", "total", "imputed"]]
    hist = hist[(hist["date"] >= base_day - pd.Timedelta(days=horizon)) &
                (hist["date"] <= base_day + pd.Timedelta(days=horizon))].assign(v=lambda x: x["total"] / UNIT)
    st.altair_chart(alt.layer(
        alt.Chart(hist).mark_line(color="black").encode(
            x=alt.X("date:T", title=None), y=alt.Y("v:Q", title=f"Saldo ({UNIT_LABEL})", scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("date:T", title="fecha"), alt.Tooltip("v:Q", format=",.2f", title="observado")]),
        alt.Chart(tot).mark_line(strokeWidth=2).encode(
            x="target_date:T", y="v:Q", color=alt.Color("método:N", title="Predicción", scale=METHOD_COLORS),
            tooltip=["método", alt.Tooltip("target_date:T", title="fecha"), alt.Tooltip("v:Q", format=",.2f")]),
        alt.Chart(pd.DataFrame({"d": [base_day]})).mark_rule(color="#888", strokeDash=[2, 2]).encode(x="d:T"),
    ), width="stretch")
    st.caption("Negro: saldo observado (si la imagen base es pasada, se ve cómo resultó). "
               "Línea gris: imagen base.")

    rows = []
    piv = fc.totals.pivot(index="target_date", columns="method", values="total")
    for hz in (1, 7, 30, 90, 180):
        t = base_day + pd.Timedelta(days=hz)
        if t not in piv.index:
            continue
        r = {"Horizonte": f"+{hz} d ({t.date()})", METHODS["single"]: piv.at[t, "single"] / UNIT,
             METHODS["multi"]: piv.at[t, "multi"] / UNIT,
             "Diferencia entre métodos": 100 * (piv.at[t, "single"] / piv.at[t, "multi"] - 1)}
        if t in h.actual_obs.index:
            act = h.actual_obs[t]
            r["Observado"] = act / UNIT
            r["Error una imagen"] = 100 * (piv.at[t, "single"] / act - 1)
            r["Error varias imágenes"] = 100 * (piv.at[t, "multi"] / act - 1)
        rows.append(r)
    if rows:
        tbl = pd.DataFrame(rows).set_index("Horizonte")
        st.dataframe(tbl, column_config={c: st.column_config.NumberColumn(format="%.2f%%" if ("Error" in c or "Diferencia" in c)
                                                                          else "%.2f") for c in tbl.columns})
        st.caption(f"Saldos en {UNIT_LABEL}; diferencias y errores en % relativo.")
    mcols = st.columns(4)
    for i, (mk, mname) in enumerate(METHODS.items()):
        mcols[2 * i].metric(f"Vencerá sin usarse · {mname}", mm(fc.meta[f"{mk}_expected_unused"]),
                            help=f"{100 * fc.meta[f'{mk}_expected_unused'] / fc.meta['balance_asof']:.1f}% del saldo de la imagen base")
        mcols[2 * i + 1].metric(f"Emisión diaria · {mname}", mm(fc.meta[f"{mk}_issuance_per_day"]))

    st.markdown("**Destino esperado del saldo de la imagen base, por fecha de vencimiento**")
    dm = st.segmented_control("Método", list(METHODS.values()), default=METHODS["multi"], key="dest")
    mkey = {v: k for k, v in METHODS.items()}[dm or METHODS["multi"]]
    b = fc.buckets[fc.buckets["method"] == mkey]
    bl = b.melt("maturity_date", ["expected_used", "expected_at_maturity"], var_name="destino", value_name="saldo")
    bl["destino"] = bl["destino"].map({"expected_used": "Se usará", "expected_at_maturity": "Vencerá sin usarse"})
    bl["saldo"] /= UNIT
    st.altair_chart(alt.Chart(bl).mark_bar().encode(
        x=alt.X("maturity_date:T", title="Fecha de vencimiento"), y=alt.Y("saldo:Q", title=f"Saldo ({UNIT_LABEL})"),
        color=alt.Color("destino:N", title=None, scale=alt.Scale(range=["#4c78a8", "#e45756"])),
        tooltip=[alt.Tooltip("maturity_date:T", title="vencimiento"), "destino", alt.Tooltip("saldo:Q", format=",.3f")]),
        width="stretch")

    st.subheader(f"¿Cómo han cambiado las predicciones guardadas? (α = {SETTINGS.alpha:g})")
    vint = h.fc_totals
    if vint["asof"].nunique() < 2:
        st.info("Se necesita más de una predicción guardada para comparar.")
    else:
        vm_label = st.segmented_control("Método ", list(METHODS.values()), default=METHODS["multi"], key="vint_m")
        vkey = {v: k for k, v in METHODS.items()}[vm_label or METHODS["multi"]]
        vint = vint[vint["method"] == vkey]
        n_show = st.slider("Predicciones a mostrar", 2, max(2, min(60, vint["asof"].nunique())),
                           min(15, vint["asof"].nunique()))
        keep = sorted(vint["asof"].unique())[-n_show:]
        vv = vint[vint["asof"].isin(keep) & (vint["target_date"] <= last_day + pd.Timedelta(days=horizon))].copy()
        vv["v"] = vv["total"] / UNIT
        vv["predicción del"] = vv["asof"].dt.strftime("%Y-%m-%d")
        act = h.actual_obs[h.actual_obs.index >= min(keep)].rename("v").rename_axis("date").reset_index()
        act["v"] /= UNIT
        st.altair_chart(alt.layer(
            alt.Chart(vv).mark_line(strokeWidth=1.2).encode(
                x=alt.X("target_date:T", title=None), y=alt.Y("v:Q", title=f"Saldo ({UNIT_LABEL})", scale=alt.Scale(zero=False)),
                color=alt.Color("predicción del:O", scale=alt.Scale(scheme="viridis")),
                tooltip=["predicción del", alt.Tooltip("target_date:T", title="fecha"), alt.Tooltip("v:Q", format=",.2f")]),
            alt.Chart(act).mark_line(color="black", strokeWidth=2.5).encode(x="date:T", y="v:Q"),
        ), width="stretch")
        st.caption("Negro: saldo observado. Colores: trayectoria prevista por cada predicción guardada.")

        rows = []
        for a_, g in vint.groupby("asof"):
            g = g.set_index("target_date")["total"]
            for hz in SETTINGS.check_horizons:
                t = a_ + pd.Timedelta(days=hz)
                if t in g.index:
                    rows.append({"asof": a_, "serie": f"+{hz} días", "v": g[t] / UNIT})
        col = f"{vkey}_expected_unused"
        if col in h.vintages:
            rows += [{"asof": a_, "serie": "Vencerá sin usarse", "v": v / UNIT}
                     for a_, v in h.vintages.set_index("asof")[col].dropna().items()]
        hz_df = pd.DataFrame(rows).sort_values(["serie", "asof"])
        hz_df["cambio_pct"] = 100 * hz_df.groupby("serie")["v"].pct_change()
        st.markdown("**Predicción a horizonte fijo según la fecha en que se hizo** (un salto = revisión significativa)")
        st.altair_chart(alt.layer(
            alt.Chart(hz_df).mark_line(point=True).encode(
                x=alt.X("asof:T", title="Fecha de la predicción"), y=alt.Y("v:Q", title=UNIT_LABEL, scale=alt.Scale(zero=False)),
                color=alt.Color("serie:N", title=None),
                tooltip=[alt.Tooltip("asof:T", title="predicción del"), "serie", alt.Tooltip("v:Q", format=",.2f"),
                         alt.Tooltip("cambio_pct:Q", format="+.2f", title="cambio % vs anterior")]),
            *alert_rules(h.alerts[h.alerts["message"].str.contains(f"[{METHODS[vkey]}]", regex=False, na=False)]),
        ), width="stretch")

        acc = forecast_accuracy(h.fc_totals, h.actual_obs)
        if not acc.empty:
            st.markdown("**Precisión histórica de las predicciones guardadas** (error relativo contra imágenes observadas)")
            acc["método"] = acc["method"].map(METHODS)
            summ = acc.groupby(["horizon_bin", "método"], observed=True)["error_pct"].agg(
                comparaciones="size", sesgo="mean", error_abs=lambda x: x.abs().mean()).reset_index()
            summ = summ.pivot(index="horizon_bin", columns="método", values=["error_abs", "sesgo"])
            summ.columns = [f"{'Error absoluto medio' if a == 'error_abs' else 'Sesgo'} · {m}" for a, m in summ.columns]
            summ.index.name = "Horizonte"
            st.dataframe(summ, column_config={c: st.column_config.NumberColumn(format="%.2f%%") for c in summ.columns})

    with st.expander("Cómo funcionan el índice y las predicciones"):
        st.markdown(f"""
* **Una imagen**: dentro de una sola imagen conviven títulos de todas las edades. El saldo según los días
  desde la emisión (vencimiento − {fc.single.tenor} días) dibuja la curva de depreciación; se suaviza con α
  (suavizado exponencial hacia adelante y hacia atrás, en logaritmos), se fuerza a ser no creciente y se normaliza a 100
  en la emisión. Supone que la emisión ha sido similar en el último año.
* **Varias imágenes**: comparando imágenes consecutivas se mide cada día qué fracción del saldo vigente se usó.
  Encadenando esas tasas se obtiene un índice en el tiempo (base 100); la tasa diaria se suaviza con α y el último
  valor suavizado es el ritmo que se proyecta, con la forma relativa por días al vencimiento de los últimos
  {SETTINGS.lookback_days} días.
* **α**: peso del dato más reciente. α pequeño = tendencia suave y memoria larga; α = 1 = sin suavizar.
  La pestaña *Índice de depreciación* permite calibrarlo con la historia.
* Cada vencimiento se proyecta día a día; al llegar su fecha, lo que quede **vence sin usarse**. Las nuevas emisiones se
  suman al promedio diario de los últimos {SETTINGS.issuance_window_days} días.
* Las predicciones guardadas (una por imagen, con α = {SETTINGS.alpha:g}) no se reescriben, para ver cómo cambian y medir
  su error.
""")

# --------------------------------------------------------------------------- tab: alerts
with tabs[5]:
    st.subheader("Registro de alertas")
    st.caption(f"Todas las alertas son relativas al valor que describen: uso o emisión diaria (% del saldo) frente a su "
               f"nivel suavizado (umbral +{SETTINGS.usage_rel_pct:.0f}% / +{SETTINGS.issuance_rel_pct:.0f}%), cambio diario del "
               f"saldo (±{SETTINGS.balance_change_pct:g}%), cambio del ritmo de depreciación suavizado (±{SETTINGS.index_change_pct:.0f}%), "
               f"revisión entre predicciones (±{SETTINGS.forecast_change_pct:g}%; saldo que vencerá sin usarse "
               f"±{SETTINGS.unused_change_pct:g}%), error de la predicción anterior "
               f"(±{SETTINGS.forecast_error_pct:g}%) y discrepancia entre métodos (±{SETTINGS.method_gap_pct:g}%). "
               "Severidad *alerta* = más del doble del umbral.")
    if h.alerts.empty:
        st.success("Sin alertas.")
    else:
        f = st.columns(3)
        cats = sorted(h.alerts["category"].unique())
        cats = f[0].multiselect("Categoría", cats, default=cats)
        sevs = f[1].multiselect("Severidad", ["alerta", "advertencia", "info"], default=["alerta", "advertencia"])
        min_rel = f[2].number_input("Cambio relativo mínimo (|%|)", min_value=0.0, value=0.0, step=5.0)
        al = h.alerts[h.alerts["category"].isin(cats) & h.alerts["severity"].isin(sevs)].copy()
        al = al[al["change_pct"].abs().fillna(np.inf) >= min_rel].sort_values("date", ascending=False)
        al["indicador"] = al["metric"].map(METRIC_LABELS).fillna(al["metric"])
        st.dataframe(al[["date", "severity", "category", "indicador", "change_pct", "message", "first_detected"]],
                     hide_index=True,
                     column_config={"date": st.column_config.DateColumn("Fecha"),
                                    "severity": st.column_config.TextColumn("Severidad"),
                                    "category": st.column_config.TextColumn("Categoría"),
                                    "indicador": st.column_config.TextColumn("Indicador"),
                                    "change_pct": st.column_config.NumberColumn("Cambio relativo", format="%+.1f%%"),
                                    "message": st.column_config.TextColumn("Detalle", width="large"),
                                    "first_detected": st.column_config.TextColumn("Detectada el")})
        st.download_button("Descargar alertas (CSV)", h.alerts.to_csv(index=False).encode("utf-8"), "alertas_tidis.csv")

# --------------------------------------------------------------------------- tab: data
with tabs[6]:
    st.subheader("Imágenes guardadas")
    tc = st.column_config.TextColumn
    st.dataframe(h.catalog.sort_values("snapshot_date", ascending=False), hide_index=True,
                 column_config={"snapshot_date": st.column_config.DateColumn("Fecha imagen"),
                                "saved_at": tc("Guardada el"), "source_cut_date": tc("Fecha de corte (archivo)"),
                                "origin": tc("Origen"), "n_rows": st.column_config.NumberColumn("Títulos"),
                                "n_imputed_rows": st.column_config.NumberColumn("Filas imputadas"),
                                "total_balance": st.column_config.NumberColumn("Saldo total", format="dollar"),
                                "sha256": None})
    imp_days = h.imputed[h.imputed].index
    st.markdown(f"**Días imputados:** {len(imp_days)}" +
                (f" — {'… ' if len(imp_days) > 15 else ''}{', '.join(str(d.date()) for d in imp_days[-15:])}"
                 if len(imp_days) else ""))
    st.caption(f"Método: títulos vigentes a ambos lados del hueco → interpolación '{SETTINGS.imputation}'; "
               "títulos que vencen en el hueco → saldo constante hasta su vencimiento; "
               "títulos nuevos → aparecen en su fecha de emisión estimada. "
               "Filas con saldo o vencimiento faltante → se completan con la imagen anterior del mismo título.")
    d = st.columns(4)
    d[0].download_button("Historia diaria (CSV)", h.flows.to_csv().encode("utf-8"), "historia_tidis.csv")
    panel_long = h.panel.stack().rename("saldo").reset_index()
    panel_long.columns = ["fecha_imagen", "fecha_vencimiento", "saldo"]
    panel_long = panel_long[panel_long["saldo"] > 0].assign(imputado=lambda x: x["fecha_imagen"].map(h.imputed))
    d[1].download_button("Imágenes diarias (CSV)", panel_long.to_csv(index=False).encode("utf-8"), "imagenes_tidis.csv")
    if h.index.kind == "index":
        d[2].download_button("Índice de depreciación (CSV)", h.index.series.to_csv().encode("utf-8"),
                             "indice_depreciacion_tidis.csv")
    raw = store.raw_path(h.obs.index[-1])
    if raw.exists():
        d[3].download_button("Archivo original más reciente", raw.read_bytes(), raw.name)
    if n_new:
        st.caption(f"{n_new} predicción(es) nuevas guardadas en esta visita.")
