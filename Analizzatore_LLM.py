import os
import re
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from datasets import load_dataset
import pandas as pd
import time
from langchain_ollama import ChatOllama

load_dotenv()

FILE_RISPOSTE = "risposte.txt"
FILE_PUNTEGGI_CSV = "punteggi_modelli.csv"
FILE_GREZZI_CSV = "punteggi_grezzi_completi.csv"

print("Inizializzazione del LLM Giudice (Llama 3.1 via OpenRouter)...")
llm_judge = ChatOllama(
    model="llama3.1", # Usa questo se hai problemi di rate limit
    temperature=0.0,
    max_retries=3,
    timeout=15
)

# Il Prompt del Giudice (Cuore della metodologia LLM-as-a-Judge)
prompt_giudice = ChatPromptTemplate.from_messages([
    ("system", "Sei un valutatore esperto e imparziale per un benchmark accademico.\n"
               "Confronta la RISPOSTA DEL MODELLO con la RISPOSTA CORRETTA (Ground Truth).\n"
               "Regole FONDAMENTALI:\n"
               "- BARRIERA LINGUISTICA: La Ground Truth è spesso in inglese, mentre la risposta del modello è in italiano. Traduci mentalmente e valuta SOLO l'equivalenza concettuale.\n"
               "- RUMORE DI FONDO: Ignora completamente saluti, traduzioni letterali, frasi introduttive (es. 'La risposta è...', 'Ecco la continuazione...') e prolissità. Cerca il concetto chiave.\n"
               "- Se il modello individua l'opzione, il valore o il significato corretto presente nella Ground Truth, consideralo corretto a prescindere da quante parole in più ha usato.\n"
               "- Valuta la correttezza semantica, logica e fattuale del nucleo della risposta.\n"
               "- Restituisci ESATTAMENTE E SOLO il numero 1 (se è corretta) o il numero 0 (se è errata).\n"
               "Non aggiungere ALCUNA altra parola, simbolo o spiegazione."),
    ("human", "DOMANDA ORIGINALE: {domanda}\n"
              "RISPOSTA CORRETTA ATTESA: {ground_truth}\n"
              "RISPOSTA DEL MODELLO: {risposta_modello}")
])

chain_valutazione = prompt_giudice | llm_judge

# =====================================================================
# 1. Recupero delle Ground Truth dai Dataset
# =====================================================================
print("Recupero le Ground Truth dai dataset originali (Seed 42)...")

gt_dict = {}

# Funzione per normalizzare il testo e facilitare il match esatto delle chiavi
def normalize_text(text):
    return re.sub(r'\s+', ' ', str(text)).strip().lower()

try:
    # GSM8K
    ds_gsm8k = load_dataset("gsm8k", "main", split="test").shuffle(seed=42).select(range(100))
    for item in ds_gsm8k:
        gt_dict[normalize_text(item['question'])] = item['answer']

    # MMLU
    ds_mmlu = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=42).select(range(100))
    lettere = ["A", "B", "C", "D"]
    for item in ds_mmlu:
        q = f"{item['question']}\nOpzioni: A) {item['choices'][0]}, B) {item['choices'][1]}, C) {item['choices'][2]}, D) {item['choices'][3]}"
        gt_dict[normalize_text(q)] = lettere[item['answer']]

    # SQuAD
    ds_squad = load_dataset("squad", split="validation").shuffle(seed=42).select(range(100))
    for item in ds_squad:
        q = f"Basandoti sul seguente contesto, rispondi alla domanda.\nContesto: {item['context']}\nDomanda: {item['question']}"
        gt_dict[normalize_text(q)] = item['answers']['text'][0]

    # HellaSwag
    ds_hellaswag = load_dataset("Rowan/hellaswag", split="validation").shuffle(seed=42).select(range(100))
    for item in ds_hellaswag:
        q = f"Completa logicamente la seguente situazione:\n{item['ctx']}"
        # La 'label' indica l'indice della conclusione corretta in 'endings'
        gt_dict[normalize_text(q)] = item['endings'][int(item['label'])]

    # HumanEval
    ds_humaneval = load_dataset("openai_humaneval", split="test").shuffle(seed=42).select(range(100))
    for item in ds_humaneval:
        q = f"Scrivi una funzione Python che completi la seguente traccia e passi i test indicati nella docstring:\n{item['prompt']}"
        gt_dict[normalize_text(q)] = item['canonical_solution']

except Exception as e:
    print(f"Errore nel caricamento dei dataset: {e}")

print(f"Ground Truth caricate: {len(gt_dict)} query uniche.")

# =====================================================================
# 2. Parsing Robusto del file risposte.txt
# =====================================================================
print(f"\nLeggo le risposte dal file '{FILE_RISPOSTE}'...")

with open(FILE_RISPOSTE, "r", encoding="utf-8") as f:
    contenuto = f.read()

pattern = re.compile(
    r"--- RUN (\d+) \| DOMANDA (\d+) \| TASK: (.*?) \| MODELLO: (.*?) ---\n"
    r"QUERY:\n(.*?)\n"
    r"RISPOSTA:\n(.*?)\n"
    r"-{50}", 
    re.DOTALL 
)

match_trovati = pattern.findall(contenuto)
risultati_da_valutare = []

for match in match_trovati:
    run, id_domanda, task, modello, query_str, risposta_str = match
    query_norm = normalize_text(query_str)
    
    soluzione_esatta = gt_dict.get(query_norm, None)
    
    if soluzione_esatta:
        risultati_da_valutare.append({
            "run": run,
            "domanda_id": id_domanda,
            "task": task.strip(),
            "modello": modello.strip(),
            "query": query_str.strip(),
            "risposta_modello": risposta_str.strip(),
            "ground_truth": soluzione_esatta
        })
    else:
        print(f"[Warning] Ground Truth non trovata per Domanda {id_domanda}, Task {task}")

print(f"Estratte con successo {len(risultati_da_valutare)} risposte valide.")

# =====================================================================
# 3. Valutazione LLM-as-a-Judge (CON SALVATAGGIO IN TEMPO REALE)
# =====================================================================
# Inizializziamo il file CSV grezzo con l'intestazione se non esiste
if not os.path.exists(FILE_GREZZI_CSV):
    df_empty = pd.DataFrame(columns=["run", "modello", "task", "domanda_id", "voto"])
    df_empty.to_csv(FILE_GREZZI_CSV, index=False)

# Leggiamo quante valutazioni sono già state fatte per ripartire da lì
try:
    df_esistente = pd.read_csv(FILE_GREZZI_CSV)
    start_idx = len(df_esistente)
except:
    start_idx = 0

print(f"\nRipresa valutazione asincrona con LLM-as-a-Judge (da {start_idx + 1}/{len(risultati_da_valutare)})...")

for idx, res in enumerate(risultati_da_valutare[start_idx:], start=start_idx):
    print(f"Valutazione {idx+1}/{len(risultati_da_valutare)} [Modello: {res['modello']} | Task: {res['task']}]...", end=" ", flush=True)
    
    voto = 0
    try:
        giudizio = chain_valutazione.invoke({
            "domanda": res['query'],
            "ground_truth": res['ground_truth'],
            "risposta_modello": res['risposta_modello']
        })
        
        voto_str = re.sub(r"\D", "", giudizio.content.strip())
        voto = 1 if voto_str and int(voto_str) >= 1 else 0
        print(f"VOTO: {voto}")
        
    except Exception as e:
        print(f"ERRORE API: {e} -> Assegnato VOTO: 0 di default per non bloccare lo script.")
    
    # Salvataggio IN TEMPO REALE sul CSV
    dati_riga = pd.DataFrame([{
        "run": res['run'],
        "modello": res['modello'],
        "task": res['task'],
        "domanda_id": res['domanda_id'],
        "voto": voto
    }])
    dati_riga.to_csv(FILE_GREZZI_CSV, mode='a', header=False, index=False)
    
    # Pausa strategica per aggirare i limiti API gratuiti (OpenRouter free limita a 20/min o simili)
    time.sleep(3)

# =====================================================================
# 4. Esportazione e Aggregazione dei Dati (SUL TOTALE STORICO)
# =====================================================================
if os.path.exists(FILE_GREZZI_CSV):
    df_completo = pd.read_csv(FILE_GREZZI_CSV)
    
    # Calcolo accuratezza per Modello e Task
    summary = df_completo.groupby(["modello", "task"])["voto"].mean() * 100
    summary = summary.reset_index()
    summary.rename(columns={"voto": "Accuratezza (%)"}, inplace=True)
    
    # Arrotondamento a 2 decimali
    summary['Accuratezza (%)'] = summary['Accuratezza (%)'].round(2)
    
    print("\n" + "="*50)
    print("RISULTATI ACCURATEZZA FINALI")
    print("="*50)
    print(summary.to_string(index=False))
    
    # Salvataggio su CSV del summary
    summary.to_csv(FILE_PUNTEGGI_CSV, index=False)
    
    print(f"\n✅ Dati aggregati salvati in '{FILE_PUNTEGGI_CSV}'.")
    print(f"✅ I dati grezzi sono stati aggiornati progressivamente in '{FILE_GREZZI_CSV}'.")
else:
    print("Nessun punteggio generato. Controlla gli errori precedenti.")