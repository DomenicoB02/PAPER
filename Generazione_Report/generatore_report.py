import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import glob
import os
import re
import seaborn as sns
import scipy.stats as stats

print("Starting Advanced Report Generator (Energy & Token Integration)...")

# =========================================================
# 1. UNIVERSAL MAPPING FUNCTION
# =========================================================
def map_model(name):
    name_str = str(name).lower()
    if 'qwen' in name_str and '14b' in name_str: return 'Qwen 2.5 (14B)'
    if 'qwen' in name_str and '7b' in name_str: return 'Qwen 2.5 (7B)'
    if 'phi' in name_str: return 'Phi-3 Mini (3B)'
    if 'mistral' in name_str: return 'Mistral (8B)'
    if '3.2' in name_str: return 'Llama 3.2 (3B)'
    if '3.1' in name_str: return 'Llama 3.1 (8B)'
    if 'gemma' in name_str: return 'Gemma 2 (27B)'
    return 'Other'

sorter = ['Llama 3.2 (3B)', 'Phi-3 Mini (3B)', 'Qwen 2.5 (7B)', 'Mistral (8B)',
          'Llama 3.1 (8B)', 'Qwen 2.5 (14B)', 'Gemma 2 (27B)']

# =========================================================
# 2. LOADING TOKEN EXTRACTOR (FROM TXT FILE)
#    FIX: header parsing with regex instead of split by '|'
#    FIX: token count by words, not by characters
# =========================================================
file_txt = glob.glob("*.txt")
dati_estratti = []

for file in file_txt:
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        contenuto = f.read()

    blocchi = contenuto.split('--- RUN ')
    for blocco in blocchi:
        if not blocco.strip():
            continue
        try:
            # Header format: "1 | DOMANDA 1 | TASK: GSM8K | MODELLO: llama3.2 ---"
            prima_riga = blocco.split('\n')[0]

            match_task = re.search(r'TASK:\s*(\S+)', prima_riga)
            match_modello = re.search(r'MODELLO:\s*([^\s\-]+)', prima_riga)

            if not match_task or not match_modello:
                continue

            task = match_task.group(1).strip()
            modello = match_modello.group(1).strip()

            if 'RISPOSTA:' not in blocco:
                continue

            testo_risposta = blocco.split('RISPOSTA:')[1].strip()

            # FIX #1: token count by words (split by spaces/newline),
            # much more accurate than len/4 for multilingual texts
            numero_token = len(testo_risposta.split())

            dati_estratti.append({
                'modello_raw': modello,
                'task': task,
                'numero_token': numero_token
            })
        except Exception:
            continue

df_risposte = pd.DataFrame(dati_estratti)
if not df_risposte.empty:
    df_risposte['Modello'] = df_risposte['modello_raw'].apply(map_model)
    # Mean of tokens by Model and Task
    # FIX #2: removed observed=False (makes no sense on non-Categorical string columns)
    df_tokens_task = df_risposte.groupby(['Modello', 'task'])['numero_token'].mean().reset_index()
else:
    print("WARNING: No .txt files found. Token calculations failed.")
    df_tokens_task = pd.DataFrame(columns=['Modello', 'task', 'numero_token'])

# =========================================================
# 3. LOADING EMISSIONS AND SCORES (FROM CSV FILES)
# =========================================================
try:
    file_emissions = ["emissions.csv"]
    df_emissions = pd.concat([pd.read_csv(f) for f in file_emissions if os.path.exists(f)],
                             ignore_index=True)
    df_emissions = df_emissions[df_emissions["project_name"] != "idle_baseline"].copy()

    # Pulizia e conversione forzata di TUTTE le colonne che devono essere numeriche
    colonne_numeriche = [
        'energy_consumed', 'duration', 'emissions', 
        'cpu_energy', 'gpu_energy', 'ram_energy'
    ]
    
    for col in colonne_numeriche:
        if col in df_emissions.columns:
            # Sostituisce le virgole con punti (se presenti) e forza in formato float
            df_emissions[col] = df_emissions[col].astype(str).str.replace(',', '.')
            df_emissions[col] = pd.to_numeric(df_emissions[col], errors='coerce')
            
    file_scores = ["punteggi_modelli.csv"]
    df_scores = pd.concat([pd.read_csv(f) for f in file_scores if os.path.exists(f)],
                          ignore_index=True)
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit()

# FIX #3: robust extraction of model and task from project_name.
# Format: Q{N}_{modello_raw}_{task} where modello_raw can contain ':'
# The task is ALWAYS the last token, the model is everything between the
# first and last underscore.
def parse_project_name(name):
    parts = name.split('_')
    if len(parts) < 3:
        return 'Unknown', 'Unknown'
    task = parts[-1]                   # last token  → e.g. "GSM8K"
    modello = '_'.join(parts[1:-1])    # from second to second-to-last → e.g. "gemma2:27b"
    return modello, task

df_emissions[['modello_raw', 'task']] = df_emissions['project_name'].apply(
    lambda x: pd.Series(parse_project_name(x))
)

df_emissions['Modello'] = df_emissions['modello_raw'].apply(map_model)
df_scores['Modello'] = df_scores['modello'].apply(map_model)

# Filter 'Other'
df_emissions = df_emissions[df_emissions['Modello'] != 'Other']
df_scores = df_scores[df_scores['Modello'] != 'Other']

# =========================================================
# 4. ROW BY ROW POWER_WATTS CALCULATION (before aggregating)
#    FIX #4: power must be calculated on each single measurement,
#    not on the ratio of the means (mean(E/t) ≠ mean(E)/mean(t))
# =========================================================

# Forza la conversione in formato numerico per evitare che pandas tratti i dati come stringhe
df_emissions['energy_consumed'] = pd.to_numeric(df_emissions['energy_consumed'], errors='coerce')
df_emissions['duration'] = pd.to_numeric(df_emissions['duration'], errors='coerce')

# Ora la matematica funzionerà correttamente senza intaccare la memoria
df_emissions['power_watts'] = (df_emissions['energy_consumed'] * 1000 * 3600) / df_emissions['duration']
# =========================================================
# 5. DATA MERGING: ENERGY vs TOKEN
# =========================================================
df_emissions_task = df_emissions.groupby(['Modello', 'task'])['energy_consumed'].mean().reset_index()

df_merged = pd.merge(df_emissions_task, df_tokens_task, on=['Modello', 'task'], how='inner')

# FIX #5: correct conversion kWh → µWh
#   1 kWh = 1_000_000_000 µWh  (not 1_000_000)
if df_merged.empty:
    print("WARNING: the Energy-Token merge is empty. "
          "Check that the tasks in the CSVs and TXTs match.")
    df_merged['energy_per_token_uWh'] = pd.Series(dtype=float)
else:
    df_merged['energy_per_token_uWh'] = (
        df_merged['energy_consumed'] * 1_000_000_000
    ) / df_merged['numero_token']

# Global mean energy per token by model
df_global_ept = df_merged.groupby('Modello')['energy_per_token_uWh'].mean().reset_index()

# =========================================================
# 6. AGGREGATION FOR THE LARGE DASHBOARD
# =========================================================
df_avg_emissions = df_emissions.groupby('Modello').agg(
    energy_consumed=('energy_consumed', 'mean'),
    duration=('duration', 'mean'),
    emissions=('emissions', 'mean'),
    cpu_energy=('cpu_energy', 'mean'),
    gpu_energy=('gpu_energy', 'mean'),
    ram_energy=('ram_energy', 'mean'),
    # FIX #4 (continued): mean of the power already calculated row by row
    power_watts=('power_watts', 'mean')
)

df_avg_scores = df_scores.groupby('Modello')['Media Accuratezza Run (%)'].mean()

# Merging of all metrics
df_avg = df_avg_emissions.join(df_avg_scores)
df_avg = df_avg.join(df_global_ept.set_index('Modello')['energy_per_token_uWh'])

# Reorder by model size
df_avg = df_avg.reindex([m for m in sorter if m in df_avg.index])
model_names = df_avg.index.tolist()

# Scatter plot parameters (bubble size proportional to billions of parameters)
param_map = {
    'Llama 3.2 (3B)': 3, 'Phi-3 Mini (3B)': 3.8, 'Qwen 2.5 (7B)': 7,
    'Mistral (8B)': 7.2, 'Llama 3.1 (8B)': 8, 'Qwen 2.5 (14B)': 14, 'Gemma 2 (27B)': 27
}
bubble_sizes = [param_map.get(m, 7) * 400 for m in model_names]

# Define available models early so subsequent graphs can use them
modelli_presenti = [m for m in sorter if m in df_avg.index]

# =========================================================
# 7. DETAILED ENERGY PER TOKEN PER TASK GRAPH
# =========================================================
print("Generating Detailed Energy per Token Graph...")

plt.style.use('seaborn-v0_8-whitegrid')
colors_list = ['#2ecc71', '#3498db', '#9b59b6', '#f1c40f', '#e67e22', '#e74c3c', '#34495e']

if not df_merged.empty:
    fig, ax = plt.subplots(figsize=(14, 7))
    tasks_unique = sorted(df_merged['task'].unique())
    x = np.arange(len(sorter))
    width = 0.15

    for i, task in enumerate(tasks_unique):
        task_data = (
            df_merged[df_merged['task'] == task]
            .set_index('Modello')
            .reindex(modelli_presenti)['energy_per_token_uWh']
        )
        offset = (i - len(tasks_unique) / 2) * width + width / 2
        ax.bar(np.arange(len(modelli_presenti)) + offset, task_data,
               width, label=task, edgecolor='black')

    ax.set_title('Energy Efficiency: Real Consumption per single generated Token (by Task)',
                 fontweight='bold', fontsize=16, pad=15)
    ax.set_ylabel('Energy per Token (µWh)', fontweight='bold', fontsize=14)
    ax.set_xticks(np.arange(len(modelli_presenti)))
    ax.set_xticklabels(modelli_presenti, rotation=30, ha='right', fontsize=12)
    ax.legend(title='Benchmark', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('Graph_8_Energy_per_Token_Task.png', dpi=300)
    plt.close()
    print("Saved: Graph_8_Energy_per_Token_Task.png")
else:
    print("SKIP Token graph: data not available.")

# =========================================================
# 7.1 GRAPH: ENERGY VS TIME WITH 95% CONFIDENCE INTERVAL
# =========================================================
print("Generating Double Bar Chart (Energy vs Time with 95% CI)...")

df_stats = df_emissions.groupby('Modello').agg(
    energy_mean=('energy_consumed', 'mean'),
    energy_std=('energy_consumed', 'std'),
    energy_count=('energy_consumed', 'count'),
    time_mean=('duration', 'mean'),
    time_std=('duration', 'std'),
    time_count=('duration', 'count')
).reindex(modelli_presenti)

df_stats['energy_ci'] = 1.96 * (df_stats['energy_std'] / np.sqrt(df_stats['energy_count']))
df_stats['time_ci'] = 1.96 * (df_stats['time_std'] / np.sqrt(df_stats['time_count']))

fig, ax1 = plt.subplots(figsize=(14, 7))

x = np.arange(len(modelli_presenti))
width = 0.35

bar1 = ax1.bar(x - width/2, df_stats['energy_mean'], width, 
               yerr=df_stats['energy_ci'], capsize=7, 
               label='Mean Energy (kWh)', color='#3498db', edgecolor='black', error_kw=dict(lw=2, capthick=2))

ax1.set_ylabel('Energy Consumed (kWh)', fontweight='bold', color='#2980b9', fontsize=14)
ax1.tick_params(axis='y', labelcolor='#2980b9')
ax1.set_xticks(x)
ax1.set_xticklabels(modelli_presenti, rotation=30, ha='right', fontsize=12)

ax2 = ax1.twinx()
bar2 = ax2.bar(x + width/2, df_stats['time_mean'], width, 
               yerr=df_stats['time_ci'], capsize=7, 
               label='Mean Time (s)', color='#e74c3c', edgecolor='black', error_kw=dict(lw=2, capthick=2))

ax2.set_ylabel('Response Time (s)', fontweight='bold', color='#c0392b', fontsize=14)
ax2.tick_params(axis='y', labelcolor='#c0392b')

ax1.set_title('Energy vs Time Comparison with 95% Confidence Interval', fontweight='bold', fontsize=16, pad=15)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.02, 0.98))

plt.tight_layout()
plt.savefig('Graph_9_Energy_Time_CI95.png', dpi=300)
plt.close()
print("Saved: Graph_9_Energy_Time_CI95.png")

# =========================================================
# 7.2 GROUPED BAR CHART: ENERGY INSTABILITY INDEX (CV%)
# =========================================================
print("Generating Bar Chart: Energy Instability Index (CV%)...")

# Calcolo del Coefficiente di Variazione (CV%) = (Deviazione Standard / Media) * 100
df_cv = df_emissions.groupby(['Modello', 'task'])['energy_consumed'].agg(['mean', 'std']).reset_index()
df_cv['CV_percent'] = (df_cv['std'] / df_cv['mean']) * 100

fig, ax = plt.subplots(figsize=(14, 7))

# Grafico a barre raggruppate per confrontare l'instabilità di ogni task per ogni modello
sns.barplot(
    data=df_cv, 
    x='Modello', 
    y='CV_percent', 
    hue='task', 
    palette='Set2', 
    edgecolor='black',
    order=modelli_presenti,
    ax=ax
)

ax.set_title('Energy Instability Index by Task (CV %)', fontweight='bold', fontsize=16, pad=15)
ax.set_ylabel('Instability (CV %)', fontweight='bold', fontsize=14)
ax.set_xlabel('Model', fontweight='bold', fontsize=14)

plt.xticks(rotation=30, ha='right', fontsize=12)

# Spostiamo la legenda fuori per non coprire i dati
ax.legend(title='Benchmark / Task', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('Graph_10_Bar_Energy_Instability.png', dpi=300)
plt.close()
print("Saved: Graph_10_Bar_Energy_Instability.png")

# =========================================================
# 7.3 HEATMAP: MODEL ACCURACY VS BENCHMARK
# =========================================================
print("Generating Heatmap: Model Accuracy vs Task...")

colonna_task = 'task' if 'task' in df_scores.columns else ('benchmark' if 'benchmark' in df_scores.columns else None)
colonna_acc = 'Media Accuratezza Run (%)' 

if colonna_task and colonna_acc in df_scores.columns:
    pivot_acc = df_scores.pivot_table(index='Modello', columns=colonna_task, values=colonna_acc, aggfunc='mean').reindex(modelli_presenti)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_acc, annot=True, fmt=".1f", cmap="viridis", linewidths=.5, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
    
    ax.set_title('Model Accuracy by Benchmark (%)', fontweight='bold', fontsize=16, pad=15)
    ax.set_ylabel('Model', fontweight='bold', fontsize=14)
    ax.set_xlabel('Benchmark / Task', fontweight='bold', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('Graph_11_Heatmap_Accuracy.png', dpi=300)
    plt.close()
    print("Saved: Graph_11_Heatmap_Accuracy.png")
else:
    print(f"WARNING: Cannot generate accuracy heatmap. "
          f"Required columns not found in df_scores (looked for '{colonna_task}' and '{colonna_acc}').")

# =========================================================
# 8. COMPLETE DASHBOARD
# =========================================================
print("Generating Complete Dashboard...")

TITOLO_SIZE = 22
LABEL_SIZE = 16
TICK_SIZE = 12

fig = plt.figure(figsize=(26, 42))
fig.patch.set_facecolor('#f8f9fa')
gs = gridspec.GridSpec(6, 4, figure=fig, height_ratios=[1, 1, 1.2, 1, 0.8, 0.8])

colors_hw = ['#3498db', '#e74c3c', '#2ecc71']
labels_hw = ['CPU', 'GPU', 'RAM']


def style_ax(ax, title, ylabel="", xlabel=""):
    ax.set_title(title, fontweight='bold', fontsize=TITOLO_SIZE, pad=15)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    if not xlabel and len(ax.get_xticklabels()) > 0:
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')


def make_pie(ax, data, title):
    sizes = [data['cpu_energy'], data['gpu_energy'], data['ram_energy']]
    if sum(sizes) == 0 or any(pd.isna(v) for v in sizes):
        ax.text(0.5, 0.5, "N/A", ha='center', fontsize=TICK_SIZE)
    else:
        ax.pie(sizes, labels=labels_hw, autopct='%1.1f%%', startangle=140,
               colors=colors_hw, textprops={'fontsize': 11, 'fontweight': 'bold'},
               pctdistance=0.75)
    ax.set_title(title, fontweight='bold', fontsize=16, pad=10)


# Row 0
ax_energy = fig.add_subplot(gs[0, 0:2])
ax_energy.bar(model_names, df_avg['energy_consumed'], color=colors_list, edgecolor='black')
style_ax(ax_energy, 'Mean Energy (kWh)', 'kWh')

ax_time = fig.add_subplot(gs[0, 2:4])
ax_time.bar(model_names, df_avg['duration'], color=colors_list, edgecolor='black')
style_ax(ax_time, 'Response Time (s)', 'Seconds')
for i, v in enumerate(df_avg['duration']):
    ax_time.text(i, v + (max(df_avg['duration']) * 0.02),
                 f"{v:.1f}s", ha='center', fontweight='bold', fontsize=11)

# Row 1
ax_emissions = fig.add_subplot(gs[1, 0:2])
ax_emissions.barh(model_names[::-1], (df_avg['emissions'] * 1000)[::-1],
                  color=colors_list[::-1], edgecolor='black')
style_ax(ax_emissions, 'Emissions (g CO2eq)', xlabel='Grams of CO2')

ax_power = fig.add_subplot(gs[1, 2:4])
bars_power = ax_power.bar(model_names, df_avg['power_watts'],
                          color=colors_list, edgecolor='black')
style_ax(ax_power, 'Absorbed Power (Watts)', 'W')
for bar in bars_power:
    yval = bar.get_height()
    ax_power.text(bar.get_x() + bar.get_width() / 2.0, yval + 1,
                  f"{yval:.0f}W", ha='center', va='bottom', fontweight='bold', fontsize=11)

# Row 2
ax_acc = fig.add_subplot(gs[2, 0:2])
bars = ax_acc.bar(model_names, df_avg['Media Accuratezza Run (%)'],
                  color=colors_list, edgecolor='black')
style_ax(ax_acc, 'Mean Task Accuracy (%)', 'Percentage (%)')
for bar in bars:
    yval = bar.get_height()
    ax_acc.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5,
                f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=12)

ax_scatter = fig.add_subplot(gs[2, 2:4])
ax_scatter.scatter(df_avg['energy_consumed'], df_avg['Media Accuratezza Run (%)'],
                   color=colors_list, s=bubble_sizes, edgecolor='black', alpha=0.7, zorder=5)
style_ax(ax_scatter, 'Trade-off: Energy vs Accuracy',
         'Accuracy (%)', 'Energy Consumed (kWh)')
for i, model in enumerate(model_names):
    ax_scatter.annotate(
        model,
        (df_avg['energy_consumed'].iloc[i], df_avg['Media Accuratezza Run (%)'].iloc[i]),
        xytext=(10, -5), textcoords='offset points', fontweight='bold', fontsize=11
    )

# Row 3: energy per token graph (if available)
ax_tokens = fig.add_subplot(gs[3, 1:3])
ept_vals = df_avg['energy_per_token_uWh']
if ept_vals.notna().any():
    bars_token = ax_tokens.bar(model_names, ept_vals, color=colors_list, edgecolor='black')
    style_ax(ax_tokens, 'Mean Energy Cost per Token', 'MicroWatt-hours (µWh)')
    max_val = ept_vals.fillna(0).max()
    for bar in bars_token:
        yval = bar.get_height()
        if pd.notna(yval) and yval > 0:
            ax_tokens.text(bar.get_x() + bar.get_width() / 2.0,
                           yval + max_val * 0.02,
                           f"{yval:.1f}", ha='center', va='bottom',
                           fontweight='bold', fontsize=11)
else:
    ax_tokens.text(0.5, 0.5, "Token data not available",
                   ha='center', va='center', transform=ax_tokens.transAxes, fontsize=14)
    style_ax(ax_tokens, 'Mean Energy Cost per Token', 'MicroWatt-hours (µWh)')

# Rows 4 and 5: HW distribution pies
for i, model in enumerate(model_names):
    row_idx = 4 if i < 4 else 5
    col_idx = i % 4
    ax_pie = fig.add_subplot(gs[row_idx, col_idx])
    make_pie(ax_pie, df_avg.loc[model], f"HW: {model}")

plt.tight_layout(pad=6.0)
plt.savefig('Report_AI_7_Modelli_Completo_Ordinato.png', dpi=300, bbox_inches='tight')
print("Operation completed successfully!")