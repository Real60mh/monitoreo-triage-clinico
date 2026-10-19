import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from model import clustering, preprocessing

st.set_page_config(
    page_title="Monitoreo y Triage Clínico",
    page_icon="🏥",
    layout="wide",
)


@st.cache_data
def cargar_datos_clinicos():
    np.random.seed(42)
    # Grupo 1: Pacientes normotensos / estables
    sanos = np.random.normal(loc=[75, 98.0, 36.5], scale=[5, 1.0, 0.3], size=(150, 3))
    # Grupo 2: Pacientes con respuesta febril y taquicardia
    fiebre = np.random.normal(loc=[115, 96.0, 38.8], scale=[8, 1.5, 0.5], size=(50, 3))
    # Grupo 3: Pacientes con desaturación / hipoxia
    hipoxia = np.random.normal(loc=[65, 88.0, 36.0], scale=[6, 2.0, 0.4], size=(50, 3))

    data = np.vstack([sanos, fiebre, hipoxia])
    data[:, 1] = np.clip(data[:, 1], 75.0, 100.0)
    np.random.shuffle(data)

    df = pd.DataFrame(
        np.round(data, 1),
        columns=["Ritmo_Cardiaco_bpm", "Saturacion_O2_%", "Temperatura_C"],
    )
    return df


def asignar_nombres_clinicos(df, features, labels):
    """Asocia cada número de grupo (cluster) con su perfil clínico real según sus promedios."""
    nombres = {}
    unique_labels = [int(u) for u in np.unique(labels) if int(u) >= 0]

    if len(unique_labels) == 3 and set(["Ritmo_Cardiaco_bpm", "Saturacion_O2_%", "Temperatura_C"]).issubset(set(features)):
        medias = {u: df.loc[labels == u, ["Ritmo_Cardiaco_bpm", "Saturacion_O2_%", "Temperatura_C"]].mean() for u in unique_labels}
        idx_hipoxia = min(unique_labels, key=lambda u: medias[u]["Saturacion_O2_%"])
        restantes = [u for u in unique_labels if u != idx_hipoxia]
        idx_fiebre = max(restantes, key=lambda u: medias[u]["Temperatura_C"])
        idx_estable = [u for u in unique_labels if u not in (idx_hipoxia, idx_fiebre)][0]

        nombres[idx_estable] = "Nivel 1: Estable (Bajo Riesgo)"
        nombres[idx_fiebre] = "Nivel 2: Alerta Febril / Taquicardia"
        nombres[idx_hipoxia] = "Nivel 3: Crítico / Hipoxia"
    else:
        for u in unique_labels:
            nombres[u] = f"Grupo Clínico {u + 1}"

    nombres[-1] = "Atípico / Observación (-1)"
    return nombres


def plot_distribucion_grupos(series_nombres, colores_map):
    conteo = series_nombres.value_counts()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    etiquetas = list(conteo.index)
    valores = list(conteo.values)
    colores = [colores_map.get(et, "steelblue") for et in etiquetas]

    bars = ax.bar(etiquetas, valores, color=colores, edgecolor="black", linewidth=0.6, width=0.55)
    ax.set_ylabel("Número de pacientes")
    ax.set_title("Distribución de pacientes por clasificación de triage")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=12, ha="right")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 2, f"{int(h)}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(valores) * 1.18)
    fig.tight_layout()
    return fig


def plot_comparacion_k(X_model, algorithm, linkage_method):
    ks = list(range(2, 9))
    sils, dbs, inertias = [], [], []
    for k in ks:
        _, lab = clustering(X_model, algorithm, n_clusters=k, linkage=linkage_method)
        mask = lab >= 0
        if np.sum(mask) >= 2 and len(np.unique(lab[mask])) >= 2:
            sils.append(silhouette_score(X_model[mask], lab[mask]))
            dbs.append(davies_bouldin_score(X_model[mask], lab[mask]))
        else:
            sils.append(np.nan)
            dbs.append(np.nan)
        km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X_model)
        inertias.append(km.inertia_)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(ks, inertias, "o-", color="tab:green")
    axes[0].set_title("Método del Codo (Inercia)")
    axes[0].set_xlabel("Número de grupos (k)")
    axes[0].set_ylabel("Inercia WCSS")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ks, sils, "o-", color="tab:blue")
    axes[1].set_title("Coeficiente de Silhouette")
    axes[1].set_xlabel("Número de grupos (k)")
    axes[1].set_ylabel("Puntaje (mayor es mejor)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ks, dbs, "o-", color="tab:orange")
    axes[2].set_title("Índice Davies-Bouldin")
    axes[2].set_xlabel("Número de grupos (k)")
    axes[2].set_ylabel("Puntaje (menor es mejor)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_silhouette_detalle(X_eval, y_eval, mapa_nombres, colores_clinicos):
    cluster_ids = np.unique(y_eval)
    sil_all = silhouette_samples(X_eval, y_eval)
    score_avg = float(np.mean(sil_all))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y_lower = 10
    cmap_fallback = plt.cm.tab10(np.linspace(0, 1, max(len(cluster_ids), 3)))

    for idx, cid in enumerate(cluster_ids):
        vals = np.sort(sil_all[y_eval == cid])
        size = vals.shape[0]
        y_upper = y_lower + size
        nombre_g = mapa_nombres.get(int(cid), f"Grupo {cid}")
        color = colores_clinicos.get(nombre_g, cmap_fallback[idx % len(cmap_fallback)])
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, facecolor=color, edgecolor=color, alpha=0.75, label=nombre_g)
        y_lower = y_upper + 10

    ax.axvline(score_avg, color="red", linestyle="--", linewidth=1.4, label=f"Media = {score_avg:.3f}")
    ax.set_title("Diagrama de Silhouette por Paciente")
    ax.set_xlabel("Coeficiente de Silhouette")
    ax.set_ylabel("Pacientes agrupados")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")
    fig.tight_layout()
    return fig


df = cargar_datos_clinicos()

st.sidebar.header("⚙️ Parámetros del Modelo")
st.sidebar.caption("Configuración del motor de agrupamiento no supervisado para signos vitales.")

algorithm = st.sidebar.selectbox(
    "Algoritmo de agrupamiento",
    ["K-Means", "K-Medoids", "Hierarchical Clustering", "DBSCAN"],
    index=0,
)

scaler_choice = st.sidebar.selectbox(
    "Normalización de variables",
    ["standard", "minmax"],
    format_func=lambda x: "StandardScaler (Z-Score)" if x == "standard" else "MinMaxScaler [0, 1]",
)

available_cols = df.columns.tolist()
features = st.sidebar.multiselect(
    "Variables fisiológicas activas",
    available_cols,
    default=available_cols,
)

if len(features) < 2:
    st.warning("Selecciona al menos 2 variables fisiológicas en el panel lateral.")
    st.stop()

n_clusters = 3
linkage = "ward"
eps = 0.5
min_samples = 5

if algorithm in ("K-Means", "K-Medoids", "Hierarchical Clustering"):
    n_clusters = st.sidebar.slider("Número de grupos de riesgo (k)", 2, 6, 3)
if algorithm == "Hierarchical Clustering":
    linkage = st.sidebar.selectbox("Método de enlace (linkage)", ["ward", "complete", "average", "single"])
if algorithm == "DBSCAN":
    eps = st.sidebar.slider("Radio de vecindad (eps)", 0.1, 3.0, 0.6, step=0.05)
    min_samples = st.sidebar.slider("Mínimo de muestras (min_samples)", 2, 25, 5)

X_encoded = preprocessing(df, features, scaler_type=scaler_choice)
X_model = X_encoded.values

model_obj, labels = clustering(
    X_model,
    algorithm,
    n_clusters=n_clusters,
    linkage=linkage,
    eps=eps,
    min_samples=min_samples,
)

mapa_nombres = asignar_nombres_clinicos(df, features, labels)
df_out = df.copy()
df_out["Grupo_Riesgo"] = [mapa_nombres[int(l)] for l in labels]

colores_clinicos = {
    "Nivel 1: Estable (Bajo Riesgo)": "#2ca02c",
    "Nivel 2: Alerta Febril / Taquicardia": "#ff7f0e",
    "Nivel 3: Crítico / Hipoxia": "#d62728",
    "Atípico / Observación (-1)": "#7f7f7f",
}

st.title("🏥 Sistema de Monitoreo y Triage Clínico con IA")
st.write(
    "Prototipo de apoyo a la decisión clínica para la clasificación automática de pacientes en urgencias "
    "a partir de signos vitales continuos (**frecuencia cardíaca**, **saturación de oxígeno** y **temperatura corporal**)."
)

conteo_actual = df_out["Grupo_Riesgo"].value_counts()
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Pacientes Monitoreados", len(df_out))
if n_clusters == 3 and algorithm != "DBSCAN" and len(features) == 3:
    col_m2.metric("🟢 Nivel 1: Estables", int(conteo_actual.get("Nivel 1: Estable (Bajo Riesgo)", 0)))
    col_m3.metric("🟠 Nivel 2: Alerta Febril", int(conteo_actual.get("Nivel 2: Alerta Febril / Taquicardia", 0)))
    col_m4.metric("🔴 Nivel 3: Crítico (Hipoxia)", int(conteo_actual.get("Nivel 3: Crítico / Hipoxia", 0)))
else:
    col_m2.metric("Grupos Detectados", len(np.unique(labels[labels >= 0])))
    col_m3.metric("Algoritmo Activo", algorithm)
    col_m4.metric("Casos Atípicos", int(np.sum(labels == -1)))

tab_simulador, tab_cohorte, tab_validacion = st.tabs(
    [
        "🩺 1. Simulador de Ingreso a Triage",
        "📊 2. Dashboard Clínico y Gráficas",
        "📈 3. Validación y Métricas del Modelo",
    ]
)

with tab_simulador:
    st.subheader("Evaluación de Nuevo Paciente en Tiempo Real")
    st.caption(
        "Ingresa los signos vitales de un paciente o selecciona un perfil clínico de prueba para observar "
        "cómo lo ubica y clasifica el modelo de agrupamiento."
    )

    if "sim_bpm" not in st.session_state:
        st.session_state.sim_bpm = 76
        st.session_state.sim_spo2 = 98.0
        st.session_state.sim_temp = 36.5

    b1, b2, b3 = st.columns(3)
    if b1.button("Cargar Perfil Estable", use_container_width=True):
        st.session_state.sim_bpm = 75
        st.session_state.sim_spo2 = 98.0
        st.session_state.sim_temp = 36.5
    if b2.button("Cargar Perfil Febril / Taquicardia", use_container_width=True):
        st.session_state.sim_bpm = 118
        st.session_state.sim_spo2 = 95.5
        st.session_state.sim_temp = 39.0
    if b3.button("Cargar Perfil Hipoxia", use_container_width=True):
        st.session_state.sim_bpm = 64
        st.session_state.sim_spo2 = 87.0
        st.session_state.sim_temp = 36.0

    col_izq, col_der = st.columns([1, 1.35], gap="large")

    with col_izq:
        bpm_val = st.slider("Frecuencia Cardíaca (bpm)", 45, 160, key="sim_bpm")
        spo2_val = st.slider("Saturación de Oxígeno SpO₂ (%)", 78.0, 100.0, step=0.5, key="sim_spo2")
        temp_val = st.slider("Temperatura Corporal (°C)", 35.0, 41.0, step=0.1, key="sim_temp")

        scaler_sim = MinMaxScaler() if scaler_choice == "minmax" else StandardScaler()
        X_full_scaled = scaler_sim.fit_transform(df[["Ritmo_Cardiaco_bpm", "Saturacion_O2_%", "Temperatura_C"]].values)
        pac_scaled = scaler_sim.transform(np.array([[bpm_val, spo2_val, temp_val]]))

        grupos_validos = [int(u) for u in np.unique(labels) if int(u) >= 0]
        distancias = {}
        for g in grupos_validos:
            centro_g = X_full_scaled[labels == g].mean(axis=0)
            distancias[g] = float(np.linalg.norm(pac_scaled[0] - centro_g))

        grupo_asignado = min(distancias, key=distancias.get)
        nombre_asignado = mapa_nombres[grupo_asignado]

        st.markdown("#### Clasificación Sugerida por el Modelo:")
        if "Crítico" in nombre_asignado:
            st.error(
                f"**{nombre_asignado}**\n\n"
                "Desaturación severa detectada. Prioridad de atención inmediata y soporte de oxígeno suplementario."
            )
        elif "Alerta" in nombre_asignado:
            st.warning(
                f"**{nombre_asignado}**\n\n"
                "Respuesta febril y frecuencia cardíaca elevada. Valoración médica prioritaria por posible cuadro infeccioso."
            )
        elif "Estable" in nombre_asignado:
            st.success(
                f"**{nombre_asignado}**\n\n"
                "Parámetros hemodinámicos y térmicos dentro de los rangos fisiológicos normales."
            )
        else:
            st.info(f"**Clasificación:** {nombre_asignado}")

    with col_der:
        eje_y_sel = st.radio(
            "Variable en el eje vertical (Y):",
            ["Saturacion_O2_%", "Temperatura_C"],
            format_func=lambda c: "Saturación de Oxígeno (%)" if c == "Saturacion_O2_%" else "Temperatura Corporal (°C)",
            horizontal=True,
        )
        y_pac = spo2_val if eje_y_sel == "Saturacion_O2_%" else temp_val

        fig_sc, ax_sc = plt.subplots(figsize=(7.5, 4.8))
        cmap_fallback = plt.cm.tab10(np.linspace(0, 1, max(len(np.unique(labels)), 3)))

        for idx, g in enumerate(np.unique(labels)):
            nombre_g = mapa_nombres[int(g)]
            sub = df_out[labels == g]
            color_g = colores_clinicos.get(nombre_g, cmap_fallback[idx % len(cmap_fallback)])
            ax_sc.scatter(
                sub["Ritmo_Cardiaco_bpm"],
                sub[eje_y_sel],
                label=nombre_g,
                color=color_g,
                alpha=0.65,
                s=42,
                edgecolors="white",
                linewidth=0.4,
            )

        ax_sc.scatter(
            [bpm_val],
            [y_pac],
            color=colores_clinicos.get(nombre_asignado, "gold"),
            marker="*",
            s=420,
            edgecolors="black",
            linewidth=1.6,
            zorder=10,
            label="⭐ Paciente Evaluado",
        )

        ax_sc.set_xlabel("Frecuencia Cardíaca (bpm)")
        ax_sc.set_ylabel("Saturación O₂ (%)" if eje_y_sel == "Saturacion_O2_%" else "Temperatura (°C)")
        ax_sc.set_title(f"Mapa de Agrupamiento ({algorithm}) y Ubicación del Paciente")
        ax_sc.grid(True, alpha=0.25, linestyle="--")
        ax_sc.legend(loc="best", fontsize=8.5)
        fig_sc.tight_layout()
        st.pyplot(fig_sc)
        plt.close(fig_sc)

with tab_cohorte:
    c_left, c_right = st.columns([1, 1.2], gap="large")

    with c_left:
        st.subheader("Distribución por Nivel de Riesgo")
        fig_dist = plot_distribucion_grupos(df_out["Grupo_Riesgo"], colores_clinicos)
        st.pyplot(fig_dist)
        plt.close(fig_dist)

        st.subheader("Promedios Fisiológicos por Grupo")
        tabla_promedios = (
            df_out.groupby("Grupo_Riesgo")[["Ritmo_Cardiaco_bpm", "Saturacion_O2_%", "Temperatura_C"]]
            .mean()
            .round(1)
            .rename(
                columns={
                    "Ritmo_Cardiaco_bpm": "Pulso Medio (bpm)",
                    "Saturacion_O2_%": "SpO₂ Media (%)",
                    "Temperatura_C": "Temp. Media (°C)",
                }
            )
        )
        st.dataframe(tabla_promedios, use_container_width=True)

    with c_right:
        st.subheader("Registro Clínico de Pacientes (n = 250)")
        filtro = st.selectbox(
            "Filtrar registros por clasificación:",
            ["Todos"] + list(df_out["Grupo_Riesgo"].unique()),
        )
        df_vista = df_out if filtro == "Todos" else df_out[df_out["Grupo_Riesgo"] == filtro]
        st.dataframe(df_vista, use_container_width=True, height=400)

    st.markdown("---")
    st.subheader("📉 Tendencia de Signos Vitales en Muestra de Monitoreo (Gráficos Interactivos)")
    n_muestra = st.slider("Cantidad de pacientes a visualizar en serie continua:", 15, 100, 40, step=5)
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.caption("Frecuencia Cardíaca (bpm) vs. Saturación de Oxígeno (%)")
        st.line_chart(df_vista[["Ritmo_Cardiaco_bpm", "Saturacion_O2_%"]].head(n_muestra))
    with g_col2:
        st.caption("Variación de Temperatura Corporal (°C)")
        st.area_chart(df_vista[["Temperatura_C"]].head(n_muestra))

with tab_validacion:
    st.subheader("Métricas de Calidad del Agrupamiento")
    mask_eval = labels >= 0
    n_unique = len(np.unique(labels[mask_eval])) if np.any(mask_eval) else 0

    if n_unique >= 2 and np.sum(mask_eval) >= 2:
        X_eval = X_model[mask_eval]
        y_eval = labels[mask_eval]
        sil = silhouette_score(X_eval, y_eval)
        db = davies_bouldin_score(X_eval, y_eval)

        v1, v2 = st.columns(2)
        v1.metric("Coeficiente de Silhouette (Separación)", f"{sil:.4f}")
        v2.metric("Índice de Davies-Bouldin (Compacidad)", f"{db:.4f}")
    else:
        st.info("Se requieren al menos 2 grupos válidos para calcular las métricas de validación.")

    if algorithm != "DBSCAN":
        st.markdown("#### 1. Análisis del Número Óptimo de Grupos (k = 2 a 8)")
        st.caption(
            "Se observa que en k = 3 se produce el punto de inflexión en la inercia (método del codo) "
            "y el valor óptimo de separación entre los perfiles clínicos."
        )
        fig_k = plot_comparacion_k(X_model, algorithm, linkage)
        st.pyplot(fig_k)
        plt.close(fig_k)

    if n_unique >= 2 and np.sum(mask_eval) >= 2:
        st.markdown("#### 2. Coeficiente de Silhouette por Paciente")
        fig_sil = plot_silhouette_detalle(X_eval, y_eval, mapa_nombres, colores_clinicos)
        st.pyplot(fig_sil)
        plt.close(fig_sil)

    if algorithm == "Hierarchical Clustering":
        st.markdown("#### 3. Dendrograma de Agrupamiento Jerárquico")
        Z = scipy_linkage(np.asarray(X_model, dtype=np.float64), method=linkage)
        fig_den, ax_den = plt.subplots(figsize=(10, 4))
        dendrogram(Z, ax=ax_den, truncate_mode="level", p=6)
        ax_den.set_title(f"Dendrograma Clínico (enlace: {linkage})")
        ax_den.set_xlabel("Subgrupos de pacientes")
        ax_den.set_ylabel("Distancia euclidiana")
        fig_den.tight_layout()
        st.pyplot(fig_den)
        plt.close(fig_den)
