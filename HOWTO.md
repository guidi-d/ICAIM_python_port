# How-To Operativo Del Port Python ICAIM

Versione inglese: `HOWTO_EN.md`

Questo file e' la guida pratica end-to-end per usare il port Python partendo da una cartella che contiene i dati `tseri`.

Se devi capire quale documento leggere:

- `README.md`
  Mappa rapida della cartella `python_port`: cosa c'e', dove stanno input, config, output e wrapper.
- `HOWTO.md`
  Procedura operativa dall'inizio alla fine: preparazione dataset, run, plot, batch, confronti e bundle.
- `CASE1_CONFIG_GUIDE.md`
  Significato dei campi JSON e mapping dai `parameter_files` MATLAB.
- `../MANUALE_REWRITE.md`
  Manuale generale del rewrite: architettura, workflow MATLAB e Python, limiti e motivazioni progettuali.

## 1. Prerequisiti

Repository completa:

- root: `/path/to/ICAIM`
- codice Python: `/path/to/ICAIM/rewrite/python_port`
- config JSON: `/path/to/ICAIM/Scenarios/casestudy/case1/python_port/config`
- dataset descriptors: `/path/to/ICAIM/Scenarios/casestudy/case1/dataset`

Setup tipico:

```bash
cd /path/to/ICAIM/rewrite/python_port
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Se vuoi usare i wrapper da qualsiasi cartella:

```bash
export PATH="/path/to/ICAIM/rewrite/python_port/bin:$PATH"
```

Nota pratica sui wrapper:

- `icaim-run`, `icaim-run-batch`, `icaim-plot-components`, `icaim-plot-station-fits`, `icaim-compare-matlab`, `icaim-compare-matlab-batch`, `icaim-compare-runs`, `icaim-select-runs`, `icaim-create-bundle` puntano a script dentro `rewrite/python_port`
- `icaim-build-dataset-files` punta a `rewrite/build_dataset_files.py` nella repository completa
- `icaim-compare-results` punta a `rewrite/compare_outputs.py` nella repository completa
- nel bundle distribuito `build_dataset_files.py` e `compare_outputs.py` vengono copiati nel root del bundle, quindi il wrapper resta la forma piu' stabile da documentare

## 2. Partire da una cartella con i dati

Assumiamo di avere una cartella con serie temporali `tseri`, per esempio:

```bash
/path/to/ICAIM/Data/resi_ATF2026
```

Per generare i file `stn_list` e `data_input`:

```bash
icaim-build-dataset-files --series-dir /path/to/ICAIM/Data/resi_ATF2026 --data-type GPS2
```

Questo comando crea di default:

- `Scenarios/casestudy/case1/dataset/stn_list/resi_ATF2026.txt`
- `Scenarios/casestudy/case1/dataset/data_input_resi_ATF2026.txt`

Opzioni disponibili:

- `--series-dir`
  Obbligatoria. Cartella con i file `tseri`.
- `--list-name`
  Nome del file `stn_list`. Default: `<nome_cartella>.txt`.
- `--data-input-name`
  Nome del file `data_input`. Default: `data_input_<nome_cartella>.txt`.
- `--unit-input`
  Unita' di deformazione da scrivere nel `data_input`. Default: `mm`.
- `--data-type`
  Tipo GPS da scrivere nel `data_input`: `GPS1`, `GPS2`, `GPS3`. Default: `GPS3`.
- `--operation`
  Campo operazione del `data_input`. Default: `decomp`.
- `--activate`
  Aggiorna anche `dataset/data_input_file.txt`.
- `--backup-active`
  Se usato insieme a `--activate`, salva una copia `.bak` del precedente `data_input_file.txt`.

Nota importante:

- `icaim-run` non usa piu' `data_input_file.txt` come default implicito
- `--activate` serve soprattutto per compatibilita' con workflow legacy o per tenere traccia del dataset attivo storico
- per le run Python devi passare `--data-input-file` solo quando la config effettiva non definisce gia' `data_input_file`
- nel port Python il `data_input` puo' usare `GPS1` per `Up`, `GPS2` per `East/North`, `GPS3` per `East/North/Up`
- il rewrite MATLAB clean resta `GPS3`-only, quindi i confronti Python/MATLAB restano al momento limitati a dataset `GPS3`

## 3. Scegliere il file di configurazione

Le configurazioni JSON stanno in:

```text
Scenarios/casestudy/case1/python_port/config
```

Esempi utili:

- `default.config.json`
- `config.case1.verify.quick.basic.json`
- `config.case1.verify.quick.advanced.json`
- `config.case1.verify.quick.srebro.json`
- `config.case1.example.json`
- `batch.case1.search.example.json`

I comandi accettano tre forme:

- path assoluto
- path relativo alla root della repository
- nome file di config esatto, con `.json` opzionale

Esempi equivalenti:

```bash
--config-file /path/to/ICAIM/Scenarios/casestudy/case1/python_port/config/config.atf2026.json
--config-file Scenarios/casestudy/case1/python_port/config/config.atf2026.json
--config-file config.atf2026
```

Per `--config-file` non usare alias opachi come `basic`: la CLI accetta solo il nome file completo del JSON.

Importante: `Scenarios/casestudy/case1/python_port/config/default.config.json` e' la configurazione base completa. Il JSON passato con `--config-file` viene applicato come override sopra quella base.

Per capire cosa modifica davvero il JSON, usa `CASE1_CONFIG_GUIDE.md`.

## 4. Eseguire una scomposizione singola

Comando consigliato quando il JSON contiene gia' `data_input_file`:

```bash
icaim-run --config-file config.atf2026
```

Se invece il JSON e' solo un override e non contiene `data_input_file`, passa il dataset esplicitamente:

```bash
icaim-run --config-file config.case1.verify.quick.basic --data-input-file resi_ATF2026
```

Per ispezionare tutti i parametri effettivi prima del calcolo:

```bash
icaim-run --config-file config.case1.verify.quick.basic --data-input-file resi_ATF2026 --config-only
```

Se vuoi anche il dump completo in stdout, aggiungi `--print-effective-config`.

Equivalente come script:

```bash
python /path/to/ICAIM/rewrite/python_port/run_decomposition.py \
  --config-file config.atf2026
```

All'avvio la CLI stampa:

- `config_file=...`
- `data_input_file=...`
- `dataset_label=...`
- `output_dir=...`
- `effective_config_file=...`

All'avvio `icaim-run` scrive anche `effective_config.json` nella directory di output, prima della decomposizione.

Queste informazioni finiscono anche in `summary.json` dentro `run_metadata`.

Se sia il JSON sia la CLI specificano il dataset, vince `--data-input-file`.

Output di default:

```text
Scenarios/casestudy/case1/python_port/output/<dataset>
```

File principali prodotti:

- `effective_config.json`
- `all_python.npz`
- `all_python.mat`
- `summary.json`

Guida dedicata al contenuto di `all_python.npz`:

- `ALL_PYTHON_NPZ_GUIDE.md`

Opzioni disponibili per `icaim-run` / `run_decomposition.py`:

- `--repo-root`
  Root della repository ICAIM.
- `--output-dir`
  Directory output della run. Default: `Scenarios/casestudy/case1/python_port/output/<dataset>`.
- `--data-input-file`
  Facoltativa se il JSON definisce `data_input_file`. Altrimenti e' obbligatoria.
- `--config-file`
  Obbligatoria. JSON di override sopra la config base nel codice; accetta il nome file completo, con `.json` opzionale.
- `--print-effective-config`
  Stampa in stdout il JSON completo della configurazione effettiva prima del calcolo.
- `--config-only`
  Risolve la configurazione effettiva, scrive `effective_config.json`, poi esce senza lanciare la decomposizione.
- `--make-plots`
  Genera anche i plot delle componenti dopo la run.
- `--plot-output-dir`
  Directory per i plot delle componenti. Default: `<output-dir>/plots`.
- `--plot-normalization`
  Una tra `peak-to-peak`, `unit-max`, `none`. Default: `peak-to-peak`.
- `--decomposition`
  Una tra `ICA`, `PCA`. Default: `ICA`.
- `--background-grid`
  Griglia GMT/NetCDF opzionale per lo sfondo mappa. Usa `auto` o `none`.
- `--components`
  Indici 1-based opzionali delle componenti da plottare quando usi `--make-plots`. Default: tutte.
- `--label-stations`
  Aggiunge i nomi stazione sulle mappe delle componenti quando usi `--make-plots`.
- `--dpi`
  Risoluzione PNG dei plot generati. Default: `200`.
- `--prefix`
  Prefisso opzionale dei file dei component plot. Default: `IC` per ICA e `PC` per PCA.
- `--make-station-fit-plots`
  Genera anche i plot osservato-vs-modellato per stazione.
- `--station-fit-output-dir`
  Directory per i plot per stazione. Default: `<output-dir>/station_fits`.
- `--hide-station-components`
  Nasconde le singole curve di contributo delle componenti nei station-fit plot.
- `--stations`
  Lista opzionale di codici stazione per gli station-fit generati dentro `icaim-run`, per esempio `--stations ANCG AT01`.

Esempio con plot inclusi:

```bash
icaim-run \
  --config-file config.atf2026 \
  --make-plots \
  --label-stations \
  --make-station-fit-plots
```

## 5. Generare i plot da un output gia' esistente

Plot delle componenti:

```bash
icaim-plot-components --results-file resi_ATF2026
```

Opzioni disponibili per `icaim-plot-components` / `plot_ica_components.py`:

- `--repo-root`
  Root della repository.
- `--results-mat`
  Alias storico per il file risultato `.npz` o `.mat`.
- `--results-file`
  File risultato, directory output o nome dataset. Esempio: `resi_ATF2026`.
- `--decomposition`
  Una tra `ICA`, `PCA`. Default: `ICA`.
- `--output-dir`
  Directory dei plot. Default: `<results-dir>/plots`.
- `--components`
  Indici 1-based delle componenti da plottare. Default: tutte.
- `--normalization`
  Una tra `peak-to-peak`, `unit-max`, `none`. Default: `peak-to-peak`.
- `--background-grid`
  Griglia di sfondo opzionale. Usa `auto` o `none`.
- `--label-stations`
  Scrive i nomi stazione sulla mappa.
- `--dpi`
  Risoluzione PNG. Default: `200`.
- `--prefix`
  Prefisso dei file output. Default: `IC` per ICA, `PC` per PCA.

Plot per stazione:

```bash
icaim-plot-station-fits --results-file resi_ATF2026
```

Opzioni disponibili per `icaim-plot-station-fits` / `plot_station_fits.py`:

- `--repo-root`
  Root della repository.
- `--results-file`
  File risultato, directory output o nome dataset.
- `--output-dir`
  Directory dei plot. Default: `<results-dir>/station_fits`.
- `--decomposition`
  Una tra `ICA`, `PCA`. Default: `ICA`.
- `--stations`
  Lista opzionale di codici stazione, per esempio `--stations ANCG AT01`.
- `--hide-components`
  Mostra solo osservato vs modellato totale.

## 6. Eseguire una campagna batch

Comando consigliato:

```bash
icaim-run-batch --batch-file example --data-input-file resi_ATF2026
```

Se il `base_config_file` del batch o gli override effettivi definiscono gia' `data_input_file`, puoi ometterla:

```bash
icaim-run-batch --batch-file batch.case1.search.atf2026_season.json
```

Equivalente come script:

```bash
python /path/to/ICAIM/rewrite/python_port/run_decomposition_batch.py \
  --batch-file batch.case1.search.example.json \
  --data-input-file resi_ATF2026
```

Output di default:

```text
Scenarios/casestudy/case1/python_port/output_batch/<batch-name>
```

Il batch:

- espande la griglia di configurazioni
- scarta le combinazioni incompatibili
- crea una cartella output per ogni run
- salva un `batch_summary.json`

Opzioni disponibili per `icaim-run-batch` / `run_decomposition_batch.py`:

- `--repo-root`
  Root della repository.
- `--batch-file`
  Obbligatoria. JSON batch con griglia e override condivisi.
- `--output-dir`
  Directory output batch. Default: `Scenarios/casestudy/case1/python_port/output_batch/<batch-name>`.
- `--data-input-file`
  Facoltativa se ogni config effettiva del batch definisce `data_input_file`. Altrimenti e' obbligatoria.
- `--make-plots`
  Genera i plot delle componenti per ogni run completata.
- `--plot-output-dir`
  Root directory per i plot componenti.
- `--plot-normalization`
  Una tra `peak-to-peak`, `unit-max`, `none`. Default: `peak-to-peak`.
- `--decomposition`
  Una tra `ICA`, `PCA`. Default: `ICA`.
- `--background-grid`
  Griglia di sfondo opzionale. Usa `auto` o `none`.
- `--make-station-fit-plots`
  Genera anche i station-fit plot per ogni run completata.
- `--station-fit-output-dir`
  Root directory per i station-fit plot.
- `--hide-station-components`
  Nasconde le singole componenti nei station-fit plot.

## 7. Confrontare Python e MATLAB

Confronto con la baseline storica `all.mat`:

```bash
icaim-compare-matlab --config-file config.case1.verify.quick.basic --data-input-file resi_ATF2026
```

Opzioni disponibili per `icaim-compare-matlab` / `compare_with_matlab.py`:

- `--repo-root`
  Root della repository.
- `--reference`
  File `.mat` di riferimento MATLAB.
- `--output-dir`
  Directory output. Default: `Scenarios/casestudy/case1/python_port/output/<dataset>`.
- `--data-input-file`
  Facoltativa se il JSON definisce `data_input_file`. Altrimenti e' obbligatoria.
- `--config-file`
  Obbligatoria. JSON di configurazione; accetta il nome file completo, con `.json` opzionale.

Nota importante:

- questo workflow passa anche dal rewrite MATLAB clean
- quindi qui il dataset deve ancora essere `GPS3`

Confronto Python vs MATLAB sulla stessa config o su un batch:

```bash
icaim-compare-matlab-batch --config-file config.case1.verify.quick.basic --data-input-file resi_ATF2026
```

Oppure:

```bash
icaim-compare-matlab-batch --batch-file example --data-input-file resi_ATF2026
```

Se invece la config singola o il batch definiscono gia' `data_input_file`, puoi ometterla:

```bash
icaim-compare-matlab-batch --batch-file batch.case1.search.atf2026_season.json
```

Opzioni disponibili per `icaim-compare-matlab-batch` / `compare_matlab_batch.py`:

- `--repo-root`
  Root della repository.
- `--config-file`
  Config JSON di una singola run.
- `--batch-file`
  JSON batch di piu' run.
- `--output-dir`
  Directory output. Default: `Scenarios/casestudy/case1/python_port/output_compare_batch/<config-or-batch-name>`.
- `--matlab-binary`
  Path all'eseguibile MATLAB.
- `--data-input-file`
  Facoltativa se ogni config effettiva definisce `data_input_file`. Altrimenti e' obbligatoria.

Nota:

- per `icaim-compare-matlab-batch` devi passare `--config-file` oppure `--batch-file`
- `--data-input-file` e' facoltativo solo se ogni config effettiva definisce `data_input_file`
- anche qui il dataset resta, per ora, limitato a `GPS3`, perche' la parte MATLAB clean non supporta ancora `GPS1/GPS2`

## 8. Confrontare run diverse tra loro

Confronto sintetico tra piu' output:

```bash
icaim-compare-runs \
  /path/to/ICAIM/Scenarios/casestudy/case1/python_port/output/resi_ATF2026/all_python.npz \
  /path/to/ICAIM/Scenarios/casestudy/case1/python_port/output_batch/batch.case1.search.example
```

Opzioni disponibili per `icaim-compare-runs` / `compare_decomposition_runs.py`:

- argomenti posizionali `inputs`
  File risultato o directory. Le directory vengono scandite ricorsivamente per `all_python.npz`.
- `--sort-by`
  Una tra `n_components`, `variance_explained_ICA`, `reduced_chi2_ICA`, `ICA_energy`, `path`.
- `--output-json`
  File JSON opzionale con il report.

Selezione automatica delle run migliori:

```bash
icaim-select-runs \
  /path/to/ICAIM/Scenarios/casestudy/case1/python_port/output_batch/batch.case1.search.example \
  --output-json /tmp/icaim_select_report.json \
  --output-markdown /tmp/icaim_select_report.md
```

Differenza pratica rispetto a `icaim-compare-runs`:

- `icaim-compare-runs` riassume e confronta tutte le run, ma non sceglie una vincitrice
- `icaim-select-runs` parte dagli stessi output e produce una raccomandazione: migliore run per ogni `n_components`, preferenza per famiglia di configurazioni e run finale consigliata

Opzioni disponibili per `icaim-select-runs` / `select_best_decomposition_runs.py`:

- argomenti posizionali `inputs`
  File risultato o directory. Le directory vengono scandite ricorsivamente per `all_python.npz`.
- `--output-json`
  File JSON opzionale con il report completo di selezione.
- `--output-markdown`
  File Markdown opzionale con il report leggibile.

Confronto puntuale tra due file:

```bash
icaim-compare-results \
  --a /path/to/run_a/all_python.npz \
  --b /path/to/run_b/all_python.npz \
  --label-a run_a \
  --label-b run_b
```

Opzioni disponibili per `icaim-compare-results` / `compare_outputs.py`:

- `--a`
  Primo file `.npz` o `.mat`.
- `--b`
  Secondo file `.npz` o `.mat`.
- `--label-a`
  Etichetta breve del primo file.
- `--label-b`
  Etichetta breve del secondo file.

## 9. Capire dove finiscono gli output

Run singola:

- `Scenarios/casestudy/case1/python_port/output/<dataset>/effective_config.json`
- `Scenarios/casestudy/case1/python_port/output/<dataset>/all_python.npz`
- `Scenarios/casestudy/case1/python_port/output/<dataset>/all_python.mat`
- `Scenarios/casestudy/case1/python_port/output/<dataset>/summary.json`

Batch:

- `Scenarios/casestudy/case1/python_port/output_batch/<batch-name>/`
- `Scenarios/casestudy/case1/python_port/output_batch/<batch-name>/batch_summary.json`

Confronto Python/MATLAB:

- `Scenarios/casestudy/case1/python_port/output_compare_batch/<config-o-batch>/`

Dentro `summary.json` trovi sempre:

- configurazione effettiva
- dataset effettivamente usato
- metriche di fit
- `run_metadata` con i path risolti stampati anche in CLI

Dentro la directory di output trovi anche sempre `effective_config.json`, scritto all'inizio della run e riutilizzabile come config esplicita completa.

## 10. Trasferire il codice su un altro PC

Nella repository completa:

1. copia tutta la repository mantenendo la struttura `Data/`, `Scenarios/`, `rewrite/`
2. ricrea il virtual environment
3. reinstalla `requirements.txt`
4. riaggiungi il `PATH` ai wrapper se vuoi usarli da qualsiasi cartella

Nel bundle:

1. entra in `bundle_case1/python_port`
2. crea il virtual environment
3. installa `requirements.txt`
4. lancia i comandi Python o i wrapper del bundle

I file `data_input` e `stn_list` con path assoluti legacy che contengono `/Data/` o `/Scenarios/` vengono rimappati automaticamente rispetto alla nuova root locale della repository o del bundle.

## 11. Creare un bundle distribuibile

Comando:

```bash
icaim-create-bundle
```

Equivalente:

```bash
python /path/to/ICAIM/rewrite/python_port/create_bundle.py
```

Opzioni disponibili per `icaim-create-bundle` / `create_bundle.py`:

- `--repo-root`
  Root della repository.
- `--output-dir`
  Directory di creazione bundle. Default: `Scenarios/casestudy/case1/python_port/bundle_case1`.

## 12. Comandi minimi da ricordare

Se hai una nuova cartella dati:

```bash
icaim-build-dataset-files --series-dir /path/to/ICAIM/Data/my_dataset
icaim-run --config-file config.case1.verify.quick.basic --data-input-file my_dataset
icaim-plot-components --results-file my_dataset
icaim-plot-station-fits --results-file my_dataset
```

Se vuoi fare una ricerca su piu' configurazioni:

```bash
icaim-run-batch --batch-file example --data-input-file my_dataset
icaim-compare-runs Scenarios/casestudy/case1/python_port/output_batch/batch.case1.search.example
icaim-select-runs Scenarios/casestudy/case1/python_port/output_batch/batch.case1.search.example
```

Se vuoi confrontare con MATLAB:

```bash
icaim-compare-matlab --config-file config.case1.verify.quick.basic --data-input-file my_dataset
icaim-compare-matlab-batch --config-file config.case1.verify.quick.basic --data-input-file my_dataset
```
