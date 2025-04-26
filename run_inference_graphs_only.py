"""
Script per eseguire l'inferenza utilizzando solo i grafi pre-costruiti, senza le immagini originali.

Uso:
    python run_inference_graphs_only.py [--model=NOME_MODELLO]

Opzioni:
    --model: Il modello da utilizzare. Opzioni: 'ChebNet', 'GAT', 'GraphSAGE'
             Se non specificato, verrà utilizzato 'ChebNet' di default.
"""
import os
import sys
import argparse
import time
import torch
import csv
import pandas as pd
from sys import platform

_base_path = '\\'.join(os.getcwd().split('\\')) + '\\' if platform == 'win32' else '/'.join(
    os.getcwd().split('/')) + '/'
sys.path.append(_base_path)

from monai.utils import set_determinism
from src.helpers.utils import get_device, get_date_time
from src.helpers.config import get_config
from src.models.gnn import GraphSAGE, GAT, ChebNet

# Parsing degli argomenti della linea di comando
parser = argparse.ArgumentParser(description='Esegui inferenza con modelli GNN pre-addestrati')
parser.add_argument('--model', type=str, default='ChebNet',
                    choices=['ChebNet', 'GAT', 'GraphSAGE'],
                    help='Modello da utilizzare (default: ChebNet)')
args = parser.parse_args()

# Configura seme per riproducibilità
set_determinism(seed=3)

# Ottieni percorsi
_config = get_config()
data_path = os.path.join(_base_path, _config.get('DATA_FOLDER'))
graph_path = os.path.join(data_path, _config.get('GRAPH_FOLDER'))
saved_path = os.path.join(_base_path, _config.get('SAVED_FOLDER'))
reports_path = os.path.join(_base_path, _config.get('REPORT_FOLDER'))
logs_path = os.path.join(_base_path, _config.get('LOG_FOLDER'))

if platform == 'win32':
    data_path = data_path.replace('/', '\\')
    graph_path = graph_path.replace('/', '\\')
    saved_path = saved_path.replace('/', '\\')
    reports_path = reports_path.replace('/', '\\')
    logs_path = logs_path.replace('/', '\\')

# Assicurati che le cartelle esistano
os.makedirs(logs_path, exist_ok=True)
log_file = os.path.join(logs_path, 'inference_graphs_only.log')
with open(log_file, 'a') as log:
    log.write(f'[{get_date_time()}] Avvio inferenza solo grafi con modello {args.model}\n')

print(f"Utilizzo del modello: {args.model}")

# Parametri del modello
num_node_features = 50  # Dimensione feature di input
num_classes = 4  # Numero di classi di output
dropout = 0.0  # Probabilità di dropout
hidden_channels = [512, 512, 512, 512, 512, 512, 512]  # Unità nascoste

# Inizializza il modello in base all'argomento
if args.model == 'ChebNet':
    k = 4  # Ordine polinomiale Chebyshev
    model = ChebNet(
        in_channels=num_node_features,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        dropout=dropout,
        K=k
    )
elif args.model == 'GAT':
    heads = 14  # Numero di attention heads
    attention_dropout = 0.1  # Dropout per l'attention
    model = GAT(
        in_channels=num_node_features,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        dropout=dropout,
        heads=heads,
        attention_dropout=attention_dropout
    )
else:  # GraphSAGE
    aggr = 'mean'  # Aggregatore
    model = GraphSAGE(
        in_channels=num_node_features,
        hidden_channels=hidden_channels,
        out_channels=num_classes,
        dropout=dropout,
        aggr=aggr
    )

# Ottieni il dispositivo (CPU/GPU)
device = get_device()
print(f"Utilizzo del dispositivo: {device}")


def find_graph_files():
    """Trova tutti i file .graph disponibili"""
    graph_files = []

    # Scorri tutte le cartelle nei grafici
    for subject_dir in os.listdir(graph_path):
        subject_path = os.path.join(graph_path, subject_dir)
        if os.path.isdir(subject_path):
            graph_file = os.path.join(subject_path, f"{subject_dir}.graph")
            if os.path.isfile(graph_file):
                graph_files.append((subject_dir, graph_file))

    # Limita a max 10 grafi per il test
    max_graphs = 10
    if len(graph_files) > max_graphs:
        print(f"Trovati {len(graph_files)} grafi. Limitando a {max_graphs} per il test.")
        graph_files = graph_files[:max_graphs]
    else:
        print(f"Trovati {len(graph_files)} grafi per il test.")

    return graph_files


def run_inference_on_graphs():
    """Esegue l'inferenza direttamente sui file dei grafi"""
    graph_files = find_graph_files()

    if not graph_files:
        print("Nessun grafo trovato. Impossibile procedere con l'inferenza.")
        return []

    # Controllo file di modello
    model_files = [f for f in os.listdir(saved_path) if args.model.upper() in f and f.endswith('_best.pth')]
    if not model_files:
        print(f"ERRORE: Nessun modello {args.model} trovato in {saved_path}")
        with open(log_file, 'a') as log:
            log.write(f'[{get_date_time()}] ERRORE: Nessun modello {args.model} trovato\n')
        return []

    latest_model = model_files[-1]
    print(f"Utilizzo del modello pre-addestrato: {latest_model}")

    # Carica il modello
    model.load_state_dict(
        torch.load(os.path.join(saved_path, latest_model), map_location=torch.device(device))
    )
    model.eval()

    # Esegui inferenza sui grafi
    results = []
    with torch.no_grad():
        for i, (subject_id, graph_file) in enumerate(graph_files):
            print(f"Inferenza sul grafo {i + 1}/{len(graph_files)}: {subject_id}")
            try:
                # Carica direttamente il grafo
                graph_data = torch.load(graph_file)

                # Sposta i tensori sul dispositivo corretto
                graph_data.x = graph_data.x.to(device)
                graph_data.edge_index = graph_data.edge_index.to(device)

                # Inferenza
                outputs = model(graph_data.x, graph_data.edge_index.type(torch.int64))
                predictions = outputs.argmax(dim=1)

                # Calcola statistiche
                class_counts = torch.bincount(predictions, minlength=num_classes)
                accuracy = (predictions == graph_data.y.to(device)).float().mean().item()

                # Salva i risultati
                results.append({
                    'subject': subject_id,
                    'class_0_count': class_counts[0].item(),
                    'class_1_count': class_counts[1].item(),
                    'class_2_count': class_counts[2].item(),
                    'class_3_count': class_counts[3].item(),
                    'accuracy': accuracy
                })

                print(f"  Accuratezza: {accuracy:.4f}")
                print(f"  Distribuzione classi: {class_counts.cpu().numpy()}")

            except Exception as e:
                print(f"  Errore durante l'inferenza per {subject_id}: {str(e)}")

    return results


try:
    # Esegui l'inferenza
    print(f"Esecuzione inferenza sui grafi disponibili...")
    start_time = time.time()

    results = run_inference_on_graphs()

    # Calcola tempo di esecuzione
    execution_time = time.time() - start_time

    # Salva i risultati in un file CSV
    if results:
        results_file = os.path.join(reports_path, f"{args.model}_graphs_inference_results.csv")
        df_results = pd.DataFrame(results)
        df_results.to_csv(results_file, index=False)
        print(f"Risultati salvati in: {results_file}")

        # Calcola metriche aggregate
        if len(results) > 0:
            avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
            total_nodes = sum(
                r['class_0_count'] + r['class_1_count'] + r['class_2_count'] + r['class_3_count'] for r in results)

            # Stampa metriche
            print("\n" + "=" * 50)
            print(f"RISULTATI INFERENZA SOLO GRAFI - MODELLO {args.model}")
            print("=" * 50)
            print(f"Tempo di esecuzione: {execution_time:.2f} secondi")
            print(f"Grafi elaborati: {len(results)}")
            print(f"Accuratezza media: {avg_accuracy:.4f}")
            print(f"Nodi totali processati: {total_nodes}")
            print("=" * 50)

            # Scrivi i risultati nel log
            with open(log_file, 'a') as log:
                log.write(f'[{get_date_time()}] Inferenza completata per {args.model}\n')
                log.write(f'[{get_date_time()}] Grafi elaborati: {len(results)}\n')
                log.write(f'[{get_date_time()}] Accuratezza media: {avg_accuracy:.4f}\n')
    else:
        print("\n" + "=" * 50)
        print(f"RISULTATI INFERENZA SOLO GRAFI - MODELLO {args.model}")
        print("=" * 50)
        print(f"Tempo di esecuzione: {execution_time:.2f} secondi")
        print(f"Nessun grafo elaborato con successo")
        print("=" * 50)

    print("Script completato con successo!")

except Exception as e:
    printgit(f"ERRORE durante l'inferenza: {str(e)}")
    with open(log_file, 'a') as log:
        log.write(f'[{get_date_time()}] ERRORE: {str(e)}\n')
    raise