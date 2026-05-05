import os
import time
import subprocess
import random
import numpy as np
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from codecarbon import EmissionsTracker

load_dotenv()

NOME_FILE_RISPOSTE = "risposte.txt"
NOME_FILE_TEMPERATURE = "temperature_log.csv"
NUM_RUNS = 1
#─────────────────────────────────────────────
# 1. CREAZIONE DEL DATASET
#─────────────────────────────────────────────
domande_test = []
RANDOM_SEED = 42

print("Carico i benchmark da Hugging Face...\n")


# MMLU (Conoscenza Generale)
dataset_mmlu = load_dataset("cais/mmlu", "all", split="test")
mmlu_sample = dataset_mmlu.shuffle(seed=RANDOM_SEED).select(range(100))
for item in mmlu_sample:
    prompt = f"{item['question']}\nOpzioni: A) {item['choices'][0]}, B) {item['choices'][1]}, C) {item['choices'][2]}, D) {item['choices'][3]}"
    domande_test.append({"task": "MMLU", "prompt": prompt})
    

# GSM8K (Matematica e Logica)
dataset_gsm8k = load_dataset("gsm8k", "main", split="test")
gsm8k_sample = dataset_gsm8k.shuffle(seed=RANDOM_SEED).select(range(100))
for item in gsm8k_sample:
    domande_test.append({"task": "GSM8K", "prompt": item['question']})

# SQuAD (Comprensione del Testo)
dataset_squad = load_dataset("squad", split="validation")
squad_sample = dataset_squad.shuffle(seed=RANDOM_SEED).select(range(100))
for item in squad_sample:
    prompt = f"Basandoti sul seguente contesto, rispondi alla domanda.\nContesto: {item['context']}\nDomanda: {item['question']}"
    domande_test.append({"task": "SQuAD", "prompt": prompt})

# HellaSwag (Ragionamento di senso comune)
dataset_hellaswag = load_dataset("Rowan/hellaswag", split="validation")
hellaswag_sample = dataset_hellaswag.shuffle(seed=RANDOM_SEED).select(range(100))
for item in hellaswag_sample:
    prompt = f"Completa logicamente la seguente situazione:\n{item['ctx']}"
    domande_test.append({"task": "HellaSwag", "prompt": prompt})

# HumanEval (Generazione di Codice Python)
dataset_humaneval = load_dataset("openai_humaneval", split="test")
humaneval_sample = dataset_humaneval.shuffle(seed=RANDOM_SEED).select(range(100))
for item in humaneval_sample:
    prompt = f"Scrivi una funzione Python che completi la seguente traccia e passi i test indicati nella docstring:\n{item['prompt']}"
    domande_test.append({"task": "HumanEval", "prompt": prompt})
  

# Mescoliamo per distribuire il carico termico
random.seed(RANDOM_SEED)
random.shuffle(domande_test)
print(f"Dataset pronto! Totale domande: {len(domande_test)}\n")

#─────────────────────────────────────────────
# 2. FUNZIONI PER TEMPERATURA E CONSUMO
#─────────────────────────────────────────────

def get_gpu_temp():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        return int(result.stdout.strip())
    except Exception as e:
        print(f"Errore lettura temperatura: {e}")
        return None
    
def misura_temperatura_baseline(campioni=6, intervallo=10):
    print("\nMisurazione temperatura baseline (60s)...")
    temperature = []
    for i in range(campioni):
        temp = get_gpu_temp()
        if temp:
            temperature.append(temp)
            print(f"  Campione {i+1}/{campioni}: {temp}°C")
        time.sleep(intervallo)
    
    baseline = np.mean(temperature)
    print(f"Temperatura baseline misurata: {baseline:.1f}°C (DevStd={np.std(temperature):.1f}°C)")
    return baseline

def cooldown(temperatura_target, timeout=300):
    print(f"Cooldown in corso (target: {temperatura_target:.1f}°C)...")
    start = time.time()
    while time.time() - start < timeout:
        temp = get_gpu_temp()
        if temp is None:
            break
        print(f"  GPU Attuale: {temp}°C", end="\r")
        if temp <= temperatura_target + 5:  # Margine di 5 gradi per evitare stalli infiniti
            print(f"\nCooldown completato in {time.time()-start:.0f}s")
            return True
        time.sleep(10)
    print(f"\nWarning: timeout cooldown superato ({timeout}s)")
    return False

#─────────────────────────────────────────────
# 3. PREPARAZIONE E SETUP
#─────────────────────────────────────────────

with open(NOME_FILE_RISPOSTE, "w", encoding="utf-8") as f:
    f.write("### BENCHMARK AI MULTI-DOMINIO ###\n")
    f.write("="*80 + "\n\n")

with open(NOME_FILE_TEMPERATURE, "w", encoding="utf-8") as f:
    f.write("run,domanda,task,modello,temp_pre,temp_post,cooldown_ok\n")

TEMPERATURA_BASELINE = misura_temperatura_baseline()

modelli_da_testare = ["llama3.2", "llama3.1", "gemma2:27b"]
parser = StrOutputParser()

#─────────────────────────────────────────────
# 4. CICLO DI INFERENZA
#─────────────────────────────────────────────

for run in range(1, NUM_RUNS+1):
    for i, item_test in enumerate(domande_test, 1):
        
        task_name = item_test["task"]
        domanda = item_test["prompt"]
        
        for nome_modello in modelli_da_testare:
            # Temperatura pre-test
            temp_pre = get_gpu_temp()
            print(f"\nRUN {run} | Q{i}/{len(domande_test)} [{task_name}] | {nome_modello} | Temp pre: {temp_pre}°C")

            llm = ChatOllama(
                model=nome_modello,
                temperature=0.6,
                num_predict=512,
                num_ctx=2048,
                repeat_penalty=1.2,
                top_p=0.95,
                top_k=20,
            )

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You must only answer what is asked of you. Answer in a structured, clear, and professional manner."),
                ("human", "{query}")
            ])

            chain = prompt_template | llm | parser

            # Tracciatore: INCLUDIAMO IL TASK NAME NEL PROGETTO!
            tracker = EmissionsTracker(
                project_name=f"Q{i}_{nome_modello}_{task_name}",
                output_dir=".",
                output_file="emissions.csv",
                log_level="error"
            )

            tracker.start()
            try:
                response = chain.invoke({"query": domanda})

                with open(NOME_FILE_RISPOSTE, "a", encoding="utf-8") as f:
                    f.write(f"--- RUN {run} | DOMANDA {i} | TASK: {task_name} | MODELLO: {nome_modello} ---\n")
                    f.write(f"QUERY:\n{domanda}\n\nRISPOSTA:\n{response}\n")
                    f.write("-" * 50 + "\n\n")

            except Exception as e:
                # Questo stampa l'errore a schermo...
                print(f"Errore con {nome_modello}: {e}")
                
                # ...E QUESTO LO SCRIVE NEL FILE DI TESTO!
                with open(NOME_FILE_RISPOSTE, "a", encoding="utf-8") as f:
                    f.write(f"--- RUN {run} | DOMANDA {i} | TASK: {task_name} | MODELLO: {nome_modello} ---\n")
                    f.write(f"QUERY:\n{domanda}\n\nRISPOSTA:\n[FALLIMENTO HARDWARE/TIMEOUT]: Il modello è crashato o ha esaurito la memoria. Errore: {e}\n")
                    f.write("-" * 50 + "\n\n")
            
            finally:
                # FONDAMENTALE: Ferma il tracciatore e scrive nel CSV!
                tracker.stop()
                dati = tracker.final_emissions_data
                if dati:
                    energia_netta = (dati.energy_consumed * 1000)
                    print(f"CO2: {dati.emissions*1000:.4f}g | Energia netta: {energia_netta:.4f} Wh | Tempo: {dati.duration:.2f}s")

            # TEMPERATURA POST-TEST
            temp_post = get_gpu_temp()

            # LOG DELLE TEMPERATURE
            with open(NOME_FILE_TEMPERATURE, "a", encoding="utf-8") as f:
                f.write(f"{run},{i},{task_name},{nome_modello},{temp_pre},{temp_post},in_attesa\n")

            # COOLDOWN (Lo facciamo dopo aver scritto il log, così non perdiamo dati se si blocca)
            ok = cooldown(TEMPERATURA_BASELINE)

print("\nBenchmark completato con successo!")