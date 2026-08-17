# ICAIM Python Port

Versione inglese: `README_EN.md`

Questa cartella contiene il port Python del caso attivo `Scenarios/casestudy/case1` e gli strumenti per eseguirlo, confrontarlo con MATLAB e analizzare run diverse tra loro.

Nota sui path:
- nella repository completa questa cartella e' `rewrite/python_port`
- nel bundle distribuito la stessa cartella e' semplicemente `python_port`

Layout operativo:
- codice, package Python, `requirements.txt` e documentazione restano in `rewrite/python_port`
- configurazioni JSON del `case1` stanno in `Scenarios/casestudy/case1/python_port/config`
- output singoli stanno in `Scenarios/casestudy/case1/python_port/output/<dataset>`
- output batch stanno in `Scenarios/casestudy/case1/python_port/output_batch/<batch-name>`
- confronti Python/MATLAB batch stanno in `Scenarios/casestudy/case1/python_port/output_compare_batch/<config-o-batch>`
- bundle distribuibile in `Scenarios/casestudy/case1/python_port/bundle_case1`

Script principali dentro `python_port`:
- `run_decomposition.py`
- `run_decomposition_batch.py`
- `compare_with_matlab.py`
- `compare_matlab_batch.py`
- `plot_ica_components.py`
- `plot_station_fits.py`
- `compare_decomposition_runs.py`
- `select_best_decomposition_runs.py`
- `create_bundle.py`

Helper di repository usati anche dai wrapper:
- `rewrite/build_dataset_files.py`
- `rewrite/compare_outputs.py`

Comandi nel `PATH`:
- `icaim-run`
- `icaim-run-batch`
- `icaim-compare-matlab`
- `icaim-compare-matlab-batch`
- `icaim-plot-components`
- `icaim-plot-station-fits`
- `icaim-compare-runs`
- `icaim-select-runs`
- `icaim-create-bundle`
- `icaim-build-dataset-files`
- `icaim-compare-results`

Equivalenze wrapper -> script:
- `icaim-run` -> `run_decomposition.py`
- `icaim-run-batch` -> `run_decomposition_batch.py`
- `icaim-compare-matlab` -> `compare_with_matlab.py`
- `icaim-compare-matlab-batch` -> `compare_matlab_batch.py`
- `icaim-plot-components` -> `plot_ica_components.py`
- `icaim-plot-station-fits` -> `plot_station_fits.py`
- `icaim-compare-runs` -> `compare_decomposition_runs.py`
- `icaim-select-runs` -> `select_best_decomposition_runs.py`
- `icaim-create-bundle` -> `create_bundle.py`
- `icaim-build-dataset-files` -> `rewrite/build_dataset_files.py` nella repository completa, `build_dataset_files.py` nel bundle
- `icaim-compare-results` -> `rewrite/compare_outputs.py` nella repository completa, `compare_outputs.py` nel bundle

Documentazione disponibile:
- guida rapida italiana: `README.md`
- how-to operativo italiano: `HOWTO.md`
- quick guide in English: `README_EN.md`
- operational how-to in English: `HOWTO_EN.md`
- guida configurazione italiana: `CASE1_CONFIG_GUIDE.md`
- configuration guide in English: `CASE1_CONFIG_GUIDE_EN.md`
- manuale generale del rewrite: `../MANUALE_REWRITE.md`
- general rewrite manual in English: `../MANUAL_REWRITE_EN.md`

Quale documento leggere:
- se vuoi solo orientarti dentro `python_port`, parti da `README.md`
- se vuoi fare davvero una scomposizione partendo da una cartella dati, parti da `HOWTO.md`
- se devi capire il significato dei campi JSON, usa `CASE1_CONFIG_GUIDE.md`
- se ti servono anche architettura, workflow MATLAB e limiti generali, usa `../MANUALE_REWRITE.md`

Stato attuale:
- loader raw `tseri`, preprocessing e selezione stazioni del `case1`
- centering `basic`
- centering `advanced` per il ramo legacy `decomp_CG_means`
- `centering.Vimposed.type`: `None`, `Heaviside`, `Linear`, `V`
- PCA `empca` e `decomp_srebro_CG_simultaneous`
- ICA `vbICA` gaussiana con `net_init='SVD'` oppure `net_init='SVD_S&J'`
- preset legacy `legacy_o1..legacy_o4` e `legacy_r1..legacy_r4`
- batch search da JSON e confronto Python vs MATLAB sulla stessa configurazione
- metriche di qualita' salvate per ogni run e confronto tra run tramite `compare_decomposition_runs.py`

Limiti attuali:
- `decompositionICA.source_type='g'`
- `decompositionICA.ICA_num=1`
- `decompositionICA.source_init='kmeans'`
- `decompositionPCA.rand_init=0`
- `centering.type='advanced'` richiede `centering.function='decomp_CG_means'`

Esecuzione rapida:

```bash
cd /path/to/ICAIM/rewrite/python_port
# nel bundle: cd /path/to/bundle_case1/python_port
python run_decomposition.py --config-file config.case1.verify.quick.basic --data-input-file resi_ATF
python run_decomposition.py --config-file config.atf2026
python run_decomposition_batch.py --batch-file batch.case1.search.example.json --data-input-file resi_ATF
python compare_with_matlab.py --config-file config.case1.verify.quick.basic --data-input-file resi_ATF
python compare_decomposition_runs.py \
  /path/to/ICAIM/Scenarios/casestudy/case1/python_port/output/resi_ATF/all_python.npz \
  /path/to/ICAIM/Scenarios/casestudy/case1/python_port/output_batch/batch.case1.search.example/*/all_python.npz
python create_bundle.py
icaim-run --config-file config.atf2026
```

Per una procedura completa dall'ingresso dati fino ai confronti e al bundle, usa `HOWTO.md`.

Configurazione:
- tutti i JSON stanno in `Scenarios/casestudy/case1/python_port/config`
- base completa: `default.config.json`
- file singolo di override: `config.case1.example.json`
- file singolo con dataset incorporato: `config.atf2026.json`
- file singolo con dataset incorporato, alias esplicito stagione: `config.atf2026_season.json`
- batch: `batch.case1.search.example.json`
- quick checks: `config.case1.verify.quick.basic.json`, `config.case1.verify.quick.advanced.json`, `config.case1.verify.quick.srebro.json`
- `default.config.json` e' la configurazione base completa del case1; gli altri `config.*.json` sono override scenario-specific applicati sopra di essa
- `--config-file` accetta path completi, path repo-relativi, oppure il nome file di config esatto con `.json` opzionale; per esempio `--config-file config.atf2026` oppure `--config-file config.case1.verify.quick.basic`. Alias brevi come `basic` non sono accettati
- `--batch-file` e `--data-input-file` accettano ancora sia path completi sia nomi brevi; per esempio `--batch-file example`, `--data-input-file resi_ATF2026`
- `run_decomposition.py` e `compare_with_matlab.py` richiedono sempre `--config-file`; `--data-input-file` e' facoltativo se la configurazione effettiva lo definisce gia', per esempio in `default.config.json` o nel JSON selezionato
- `run_decomposition_batch.py` e `compare_matlab_batch.py` accettano di omettere `--data-input-file` se ogni config effettiva del batch eredita o definisce `data_input_file`
- `default.config.json` punta di default a `Scenarios/casestudy/case1/dataset/data_input_file.txt`, ma qualunque `config.*.json` selezionato o `--data-input-file` CLI puo' sovrascriverlo
- la guida operativa per i parametri e' `CASE1_CONFIG_GUIDE.md`
- per vedere tutti i parametri effettivi prima del calcolo usa `icaim-run --config-file ... [--data-input-file ...] --config-only`
- durante una run normale `icaim-run` scrive subito `effective_config.json` dentro la directory di output, prima della decomposizione
- `default.config.json` espone anche `flags`, `outliers`, `velocity` e i campi legacy di centering `offsets_epoch_imposed` e `Ustart/Sstart/Vstart`
- i blocchi `flags` e `outliers` restano configurazione di compatibilita'; `velocity` e' mantenuto soprattutto per il percorso MATLAB clean/legacy quando si abilita il detrending, ma non cambia il flusso principale di `run_decomposition`

Portabilita' su un altro PC:
- mantieni la struttura della repository con le cartelle `Data/` e `Scenarios/`
- i JSON di esempio usano nomi file espliciti o path repo-relativi; non serve riscriverli quando cambia la root locale
- i file `data_input` e `stn_list` possono anche contenere vecchi path assoluti con segmenti `/Data/` o `/Scenarios/`: il port Python li rimappa automaticamente sulla root del repository corrente
- se introduci path assoluti esterni alla repository, quelli vanno aggiornati manualmente
- se vuoi usare i wrapper `icaim-...` da qualsiasi cartella, aggiungi `export PATH="/path/to/ICAIM/rewrite/python_port/bin:$PATH"` al tuo `~/.bashrc` o `~/.zshrc`, poi riapri la shell

Metriche e confronto tra scomposizioni:
- `effective_config.json` salva la configurazione effettiva completa gia' all'avvio della run
- `summary.json` salva `config`, `resolved_config`, `config_notes`, `metrics` e `quality_metrics`
- `all_python.npz` salva anche metriche flat, `ICA.net.energy`, `ICA.net.alphas`, `quality_metrics_json` e la config in forma JSON
- `decomposition_mode` controlla l'orientamento della scomposizione: `t` e' il default storico, `s` usa internamente la matrice trasposta ed e' selezionabile anche con `--decomposition-mode s`
- guida dettagliata delle variabili e del layout `GPS1`/`GPS2`/`GPS3`: `ALL_PYTHON_NPZ_GUIDE.md`
- `compare_decomposition_runs.py` confronta piu' file `all_python.npz` o `.mat`, stampa una tabella sintetica e calcola anche F-test tra ordini di modello compatibili
- `select_best_decomposition_runs.py` usa metriche di fit, F-test e diagnostica ARD per raccomandare la run migliore per ogni `n_components` e una scelta finale
- tra le metriche piu' utili oggi ci sono `variance_explained_*`, `reduced_chi2_*`, `weighted_rms_*`, `ICA_energy` e `ARD_ratio`

Strumenti principali:
- `run_decomposition.py`: run singola, con opzioni per grafici IC e plot osservato-vs-modellato
- `run_decomposition_batch.py`: campagne multi-configurazione da JSON
- `plot_ica_components.py`: grafici delle componenti ICA o PCA da un output esistente
- `plot_station_fits.py`: confronto per stazione tra serie osservata e ricostruzione del modello
- `compare_with_matlab.py`: confronto con `all.mat`
- `compare_decomposition_runs.py`: confronto tra piu' decomposizioni Python
- `select_best_decomposition_runs.py`: selezione automatica delle run migliori da un insieme di output
- `compare_matlab_batch.py`: confronto Python vs MATLAB sulla stessa config JSON
- `build_dataset_files.py`: generazione di `stn_list` e `data_input`
- `create_bundle.py`: bundle distribuibile

Per l'elenco completo delle opzioni:

```bash
python run_decomposition.py --help
python run_decomposition_batch.py --help
python plot_ica_components.py --help
python plot_station_fits.py --help
python compare_with_matlab.py --help
python compare_decomposition_runs.py --help
python select_best_decomposition_runs.py --help
python compare_matlab_batch.py --help
```

Verifica attuale:
- quick `basic`: allineamento quasi a precisione macchina, `ICA_ts_diff=3.10e-07`
- quick `advanced + decomp_CG_means`: ancora quasi identico, `ICA_ts_diff=1.65e-07`
- quick `decomp_srebro_CG_simultaneous + SVD_S&J`: non bit-identico, ma `var_explained_*` entro circa `1e-5`

Output principali:
- `Scenarios/casestudy/case1/python_port/output/<dataset>/all_python.npz`
- `Scenarios/casestudy/case1/python_port/output/<dataset>/all_python.mat`
- `Scenarios/casestudy/case1/python_port/output/<dataset>/summary.json`
- `Scenarios/casestudy/case1/python_port/output/<dataset>/comparison_with_matlab.json`
- `Scenarios/casestudy/case1/python_port/output/<dataset>/comparison_with_matlab.txt`
- `Scenarios/casestudy/case1/python_port/output_batch/<batch-name>/*/summary.json`
- `Scenarios/casestudy/case1/python_port/bundle_case1`
