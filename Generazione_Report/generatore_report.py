import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 1. Caricamento Dati (Unione automatica dei file multipli)
try:
    # Carica e concatena i file delle emissioni
    file_emissions = ["1_emissions.csv", "2_emissions.csv", "3_emissions.csv"]
    df_emissions = pd.concat([pd.read_csv(f) for f in file_emissions])
    df_emissions = df_emissions[df_emissions["project_name"] != "idle_baseline"].copy()
    
    # Carica e concatena i file dei punteggi
    file_scores = ["1_punteggi_modelli.csv", "2_punteggi_modelli.csv", "3_punteggi_modelli.csv"]
    df_scores = pd.concat([pd.read_csv(f) for f in file_scores])
except Exception as e:
    print(f"Errore nel caricamento dei file: {e}")
    exit()

# 2. Mappatura completa dei 7 modelli
def map_model(name):
    name_str = str(name).lower()
    if 'qwen2.5:14b' in name_str: return 'Qwen2.5 (14B)'
    if 'qwen2.5:7b' in name_str: return 'Qwen2.5 (7B)'
    if 'phi3:mini' in name_str: return 'Phi-3 Mini'
    if 'mistral' in name_str: return 'Mistral'
    if '3.2' in name_str: return 'Llama3.2 (3B)'
    if '3.1' in name_str: return 'Llama3.1 (8B)'
    if 'gemma2' in name_str or '27b' in name_str: return 'Gemma2 (27B)'
    return 'Altro'

df_emissions['Categoria'] = df_emissions['project_name'].map(map_model)
df_scores['Categoria'] = df_scores['modello'].map(map_model)

# Filtra eventuali 'Altro' non identificati se presenti
df_emissions = df_emissions[df_emissions['Categoria'] != 'Altro']
df_scores = df_scores[df_scores['Categoria'] != 'Altro']

# 3. Calcolo Medie e Join tra Emissioni e Punteggi
df_avg_emissions = df_emissions.groupby('Categoria').agg({
    'energy_consumed': 'mean',
    'duration': 'mean',
    'emissions': 'mean',
    'cpu_energy': 'mean',
    'gpu_energy': 'mean',
    'ram_energy': 'mean'
})
# Potenza in watt = J / s -> J = kWh * 1000 * 3600
df_avg_emissions['power_watts'] = (df_avg_emissions['energy_consumed'] * 1000 * 3600) / df_avg_emissions['duration']

# Media dell'accuratezza
df_avg_scores = df_scores.groupby('Categoria')['Media Accuratezza Run (%)'].mean()

# Unione finale (ordinata per accuratezza decrescente)
df_avg = df_avg_emissions.join(df_avg_scores).sort_values(by='Media Accuratezza Run (%)', ascending=False)
model_names = df_avg.index.tolist()

# --- CONFIGURAZIONE ESTETICA ---
TITOLO_SIZE = 22
LABEL_SIZE = 16
TICK_SIZE = 12 
plt.style.use('seaborn-v0_8-whitegrid')

# Creiamo una griglia 5x4 (più spazio per 7 torte e 2 grafici nuovi)
fig = plt.figure(figsize=(26, 36)) 
fig.patch.set_facecolor('#f8f9fa')
gs = gridspec.GridSpec(5, 4, figure=fig, height_ratios=[1, 1, 1.2, 0.8, 0.8])

# Nuova palette di 7 colori per i modelli
colors_list = ['#2ecc71', '#3498db', '#9b59b6', '#f1c40f', '#e67e22', '#e74c3c', '#34495e']
colors_hw = ['#3498db', '#e74c3c', '#2ecc71']
labels_hw = ['CPU', 'GPU', 'RAM']

def style_ax(ax, title, ylabel="", xlabel=""):
    ax.set_title(title, fontweight='bold', fontsize=TITOLO_SIZE, pad=15)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    # Ruota i tick x se sono nomi dei modelli (per leggibilità)
    if not xlabel and len(ax.get_xticklabels()) > 0:
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right')

def make_pie(ax, data, title):
    sizes = [data['cpu_energy'], data['gpu_energy'], data['ram_energy']]
    if sum(sizes) == 0 or pd.isna(sum(sizes)): 
        ax.text(0.5, 0.5, "N/D", ha='center', fontsize=TICK_SIZE)
    else:
        ax.pie(sizes, labels=labels_hw, autopct='%1.1f%%', startangle=140, 
               colors=colors_hw, textprops={'fontsize': 11, 'fontweight': 'bold'},
               pctdistance=0.75)
    ax.set_title(title, fontweight='bold', fontsize=16, pad=10)

# --- RIGA 0: ENERGIA E TEMPO ---
ax_energy = fig.add_subplot(gs[0, 0:2])
ax_energy.bar(model_names, df_avg['energy_consumed'], color=colors_list, edgecolor='black')
style_ax(ax_energy, 'Energia Media (kWh)', 'kWh')

ax_time = fig.add_subplot(gs[0, 2:4])
ax_time.bar(model_names, df_avg['duration'], color=colors_list, edgecolor='black')
style_ax(ax_time, 'Tempo di Risposta (s)', 'Secondi')
for i, v in enumerate(df_avg['duration']):
    ax_time.text(i, v + (max(df_avg['duration'])*0.02), f"{v:.1f}s", ha='center', fontweight='bold', fontsize=11)

# --- RIGA 1: EMISSIONI E POTENZA ---
ax_emissions = fig.add_subplot(gs[1, 0:2])
ax_emissions.barh(model_names, df_avg['emissions'] * 1000, color=colors_list, edgecolor='black')
style_ax(ax_emissions, 'Emissioni (g CO2eq)', xlabel='Grammi CO2')
ax_emissions.invert_yaxis() # Modelli con score più alto in alto

ax_power = fig.add_subplot(gs[1, 2:4])
ax_power.plot(model_names, df_avg['power_watts'], color='#8e44ad', marker='o', markersize=12, linewidth=4)
style_ax(ax_power, 'Potenza Assorbita (Watt)', 'W')

# --- RIGA 2: PUNTEGGI ACCURATEZZA E SCATTER PLOT CONFRONTO ---
ax_acc = fig.add_subplot(gs[2, 0:2])
bars = ax_acc.bar(model_names, df_avg['Media Accuratezza Run (%)'], color=colors_list, edgecolor='black')
style_ax(ax_acc, 'Accuratezza Media sui Task (%)', 'Percentuale (%)')
for bar in bars:
    yval = bar.get_height()
    ax_acc.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=12)

ax_scatter = fig.add_subplot(gs[2, 2:4])
ax_scatter.scatter(df_avg['energy_consumed'], df_avg['Media Accuratezza Run (%)'], color=colors_list, s=300, edgecolor='black', zorder=5)
style_ax(ax_scatter, 'Trade-off: Energia vs Accuratezza', 'Accuratezza (%)', 'Energia Consumata (kWh)')
for i, model in enumerate(model_names):
    ax_scatter.annotate(model, 
                        (df_avg['energy_consumed'].iloc[i], df_avg['Media Accuratezza Run (%)'].iloc[i]),
                        xytext=(10, -5), textcoords='offset points', 
                        fontweight='bold', fontsize=11)

# --- RIGHE 3 e 4: I 7 GRAFICI A TORTA ---
for i, model in enumerate(model_names):
    row_idx = 3 if i < 4 else 4
    col_idx = i % 4
    ax_pie = fig.add_subplot(gs[row_idx, col_idx])
    make_pie(ax_pie, df_avg.loc[model], f"HW: {model}")

plt.tight_layout(pad=6.0)
plt.savefig('Report_AI_7_Modelli_Completo.png', dpi=300, bbox_inches='tight')
print("Report generato con successo: 'Report_AI_7_Modelli_Completo.png'")