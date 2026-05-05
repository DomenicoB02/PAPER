import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 1. Caricamento dati
try:
    df = pd.read_csv("emissions.csv")
    df = df[df["project_name"] != "idle_baseline"].copy()
except Exception as e:
    print(f"Errore nel caricamento del file: {e}")
    exit()

# 2. Mappatura modelli
def map_model(name):
    name_str = str(name).lower()
    if 'qwen' in name_str: return 'qwen2.5:14B'
    #if '3.1' in name_str: return 'Llama3.1 (8B)'
    #if '27b' in name_str: return 'Gemma2 (27B)'
    return 'Altro'

df['Categoria'] = df['project_name'].map(map_model)

# 3. Calcolo Medie
df_avg = df.groupby('Categoria').agg({
    'energy_consumed': 'mean',
    'duration': 'mean',
    'emissions': 'mean',
    'cpu_energy': 'mean',
    'gpu_energy': 'mean',
    'ram_energy': 'mean'
}).reindex(['qwen2.5:14B'])

df_avg['power_watts'] = (df_avg['energy_consumed'] * 1000 * 3600) / df_avg['duration']

# --- CONFIGURAZIONE ESTETICA ---
TITOLO_SIZE = 22
LABEL_SIZE = 18
TICK_SIZE = 14 # Leggermente ridotto per far stare 4 torte
plt.style.use('seaborn-v0_8-whitegrid')

fig = plt.figure(figsize=(24, 28)) 
fig.patch.set_facecolor('#f8f9fa')

# Nuova griglia: 3 righe x 4 colonne
gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.8])

colors_list = ['#f1c40f', '#3498db', '#e74c3c', '#2ecc71']
colors_hw = ['#3498db', '#e74c3c', '#2ecc71']
labels_hw = ['CPU', 'GPU', 'RAM']

def style_ax(ax, title, ylabel="", xlabel=""):
    ax.set_title(title, fontweight='bold', fontsize=TITOLO_SIZE, pad=20)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

def make_pie(ax, data, title):
    sizes = [data['cpu_energy'], data['gpu_energy'], data['ram_energy']]
    if sum(sizes) == 0: 
        ax.text(0.5, 0.5, "N/D", ha='center', fontsize=TICK_SIZE)
    else:
        ax.pie(sizes, labels=labels_hw, autopct='%1.1f%%', startangle=140, 
               colors=colors_hw, textprops={'fontsize': 12, 'fontweight': 'bold'},
               pctdistance=0.75)
    ax.set_title(title, fontweight='bold', fontsize=16, pad=10)

# --- RIGA 0: ENERGIA E TEMPO (2 colonne ciascuno) ---
ax_energy = fig.add_subplot(gs[0, 0:2])
ax_energy.bar(df_avg.index, df_avg['energy_consumed'], color=colors_list, edgecolor='black')
style_ax(ax_energy, 'Energia Media (kWh)', 'kWh')

ax_time = fig.add_subplot(gs[0, 2:4])
ax_time.bar(df_avg.index, df_avg['duration'], color=colors_list, edgecolor='black')
style_ax(ax_time, 'Tempo di Risposta (s)', 'Secondi')
for i, v in enumerate(df_avg['duration']):
    ax_time.text(i, v + (v*0.02), f"{v:.1f}s", ha='center', fontweight='bold', fontsize=TICK_SIZE)

# --- RIGA 1: EMISSIONI E POTENZA (2 colonne ciascuno) ---
ax_emissions = fig.add_subplot(gs[1, 0:2])
ax_emissions.barh(df_avg.index, df_avg['emissions'] * 1000, color=colors_list, edgecolor='black')
style_ax(ax_emissions, 'Emissioni (g CO2eq)', xlabel='Grammi CO2')
ax_emissions.invert_yaxis()

ax_power = fig.add_subplot(gs[1, 2:4])
ax_power.plot(df_avg.index, df_avg['power_watts'], color='#8e44ad', marker='o', markersize=15, linewidth=5)
style_ax(ax_power, 'Potenza Assorbita (Watt)', 'W')

# --- RIGA 2: I 4 GRAFICI A TORTA (1 colonna ciascuno) ---
make_pie(fig.add_subplot(gs[2, 0]), df_avg.loc['qwen2.5:14B'], "HW: qwen2.5:14B")
#make_pie(fig.add_subplot(gs[2, 1]), df_avg.loc['Llama3.1 (8B)'], "HW: Llama3.1 8B")
#make_pie(fig.add_subplot(gs[2, 2]), df_avg.loc['Gemma2 (27B)'], "HW: Gemma2 27B")

plt.tight_layout(pad=6.0)
plt.savefig('Report_Energia_AI_Completo.png', dpi=300, bbox_inches='tight')
print("Report generato con successo: 'Report_Energia_AI_Completo.png'")