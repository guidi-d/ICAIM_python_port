# Case1 Config Guide

Versione inglese: [CASE1_CONFIG_GUIDE_EN.md](CASE1_CONFIG_GUIDE_EN.md)

Questo file accompagna la configurazione JSON del `case1`, con base completa in [default.config.json](../../Scenarios/casestudy/case1/python_port/config/default.config.json) e override di esempio in [config.case1.example.json](../../Scenarios/casestudy/case1/python_port/config/config.case1.example.json), e mappa i `parameter_files` MATLAB del `case1` verso il port Python.

## Prima di tutto

- [default.config.json](../../Scenarios/casestudy/case1/python_port/config/default.config.json) e' la configurazione base completa del case1.
- Gli altri `config.*.json` nello stesso folder sono override scenario-specific applicati sopra `default.config.json`.
- [config.case1.example.json](../../Scenarios/casestudy/case1/python_port/config/config.case1.example.json) e' un override di esempio piccolo, non un dump completo dei default.
- Per vedere tutti i parametri effettivi prima della run usa `icaim-run --config-file ... [--data-input-file ...] --config-only`.
- Una run normale scrive subito `effective_config.json` nella directory di output e salva anche `resolved_config` dentro `summary.json`.

## Se Parti Da Zero

Se non sai quali opzioni scegliere, parti da qui:

- `centering.type='basic'`
- `centering.function='empca'`
- `decompositionPCA.decomp_fcn='empca'`
- `decompositionICA.net_init='SVD'`
- `decompositionICA.n_mixed_pdfs=4`
- `decomposition_mode='t'`

Usa invece queste varianti solo in casi specifici:

- `decomposition_mode='s'` se vuoi decomporre `Xd_ts.T`, utile quando `n_series` e' molto piu' grande di `n_epochs`.
- `centering.type='advanced'` solo se vuoi riprodurre il ramo legacy con media/decomposizione congiunta.
- `centering.function='decomp_CG_means'` solo insieme a `centering.type='advanced'`.
- `decompositionPCA.decomp_fcn='decomp_srebro_CG_simultaneous'` solo se vuoi confrontarti con il ramo legacy simultaneo e accetti tempi maggiori.
- `decompositionICA.net_init='SVD_S&J'` come alternativa di ricerca, non come primo default.

Override minimo consigliato:

```json
{
  "data_input_file": "data_input_resi_ATF.txt",
  "n_components": 3,
  "decomposition_mode": "t",
  "centering": {
    "type": "basic",
    "function": "empca"
  },
  "decompositionPCA": {
    "decomp_fcn": "empca"
  },
  "decompositionICA": {
    "net_init": "SVD"
  }
}
```

## Come Viene Risolta La Configurazione

Ordine di applicazione:

1. `default.config.json` costruisce la base.
2. Il JSON passato con `--config-file` applica gli override.
3. Se passi anche `--data-input-file`, la CLI vince sul JSON.
4. La validazione normalizza alias legacy e controlla i vincoli.

Conseguenze pratiche:

- `decomp_empca` e' accettato come alias legacy, ma viene normalizzato a `empca`.
- `decomposition_mode` accetta `t`, `t-mode`, `temporal`, `time`, `s`, `s-mode`, `spatial`, `space`; nei risultati viene salvato anche `decomposition_mode_resolved`.
- chiavi sconosciute nel JSON fanno fallire il parsing
- `repo_root` e `case_dir` sono campi derivati e non si possono overrideare
- per i campi tuple-like devi usare array JSON
- per alcuni campi ICA puoi passare un singolo valore oppure un array di lunghezza `1` o `n_components`
- in `icaim-run` e `compare_with_matlab.py`, `--data-input-file` e' un override opzionale: se non lo passi, il valore puo' arrivare da `default.config.json` oppure dal JSON selezionato con `--config-file`

## Convenzioni Path

- I JSON stanno in `Scenarios/casestudy/case1/python_port/config`.
- `data_input_file` puo' essere un path completo, un path relativo alla repository, oppure solo il nome del file, per esempio `data_input_resi_ATF.txt`.
- Per `--config-file` passa il nome file esatto del JSON, con `.json` opzionale, per esempio `config.atf2026` o `config.case1.verify.quick.basic`.
- Alias brevi come `basic` non sono accettati per `--config-file`.
- `--batch-file` e `--data-input-file` accettano ancora anche nomi brevi.
- In `icaim-run-batch` e `icaim-compare-matlab-batch`, `--data-input-file` e' facoltativo se ogni config effettiva del batch eredita o definisce `data_input_file`; se sia JSON sia CLI lo specificano, vince la CLI.
- Se la repository viene spostata su un altro PC, i path repo-relativi e i vecchi path assoluti che contengono `/Data/`, `/Scenarios/` o `/rewrite/` vengono risolti rispetto alla nuova root locale.

## Mapping Dai parameter_files MATLAB

- `scen_parameters.m`
  Chiavi JSON: `first_epoch`, `last_epoch`, `n_components`, `threshold_ts_missingdata`, `threshold_epochs_missingdata`, `skip_epochs`, `unit_output`, `select_origin_lon`, `select_origin_lat`, `select_radius_km`, `velocity.file`, `velocity.format`.
- `centering_parameters.m`
  Chiavi JSON: `centering.type`, `centering.function`, `centering.iter_max`, `centering.tol`, `centering.func`, `centering.dfunc`, `centering.Vimposed.*`, `centering.offsets_epoch_imposed`, `centering.Ustart`, `centering.Sstart`, `centering.Vstart`.
- `decompositionPCA_parameters.m`
  Chiavi JSON: `decompositionPCA.decomp_fcn`, `decompositionPCA.iter_max_decomp`, `decompositionPCA.tol_decomp`, `decompositionPCA.rand_missingdata`, `decompositionPCA.rand_init`.
- `decompositionICA_parameters.m`
  Chiavi JSON: `decompositionICA.*`, inclusi `mix.*`, `noise.*`, `source.*` e i preset legacy opzionali.
- `flags_parameters.m`
  Chiavi JSON: `flags.*`.
- `outliers_parameters.m`, `center_parameters_outliers.m`, `decomposition_parameters_outliers.m`
  Chiavi JSON: `outliers.*`, `outliers.centering.*`, `outliers.decompositionPCA.*`.
- `seismicity_parameters.m`
  Nessun mapping attivo nel port Python corrente: il blocco `seismicity` e' stato rimosso perche' non usato dal workflow attivo.
- `plot_parameters.m`
  Nessun mapping attivo nel port Python corrente: il blocco `plot` e' stato rimosso perche' non usato dal workflow attivo.

## Reference Rapida Delle Sezioni Top-level

Default effettivi del codice:

- `data_input_file`
  La CLI richiede che sia definito nel JSON oppure passato con `--data-input-file`. A livello di API interna esiste un fallback storico a `Scenarios/casestudy/case1/dataset/data_input_file.txt`, ma `icaim-run` e `compare_with_matlab.py` lo disabilitano esplicitamente.
- `first_epoch`
  Default: `2010.0`.
- `last_epoch`
  Default: `2019.26164336`.
- `n_components`
  Default: `2`.
- `unit_output`
  Default: `mm`.
- `skip_epochs`
  Default: `[]`.
- `threshold_ts_missingdata`
  Default: `80.0`.
- `threshold_epochs_missingdata`
  Default: `100.0`.
- `select_origin_lon`
  Default: `12.0151`.
- `select_origin_lat`
  Default: `45.9753`.
- `select_radius_km`
  Default: `95000.0`.

Uso pratico:

- Modifica prima `centering`, `decompositionPCA`, `decompositionICA`.
- Modifica `first_epoch`, `last_epoch`, `n_components`, `skip_epochs`, soglie di missing data e selezione geografica quando stai cambiando il dataset o la finestra di analisi.
- `flags` e `outliers` restano sezioni di compatibilita' ma oggi non guidano ancora il flusso numerico principale.
- `velocity` non cambia il flusso principale di `run_decomposition`, ma resta disponibile soprattutto per il percorso MATLAB clean/legacy se si abilita il detrending.

## Reference Di centering

Default effettivi:

- `type='basic'`
- `function='empca'`
- `iter_max=1000000`
- `tol=1e-7`
- `func='func_mean_zero_sum_V_transform_corrected'`
- `dfunc='dfunc_mean_zero_sum_V_transform_corrected'`
- `offsets_epoch_imposed=[]`
- `Vimposed.type='None'`
- `Vimposed.param=[]`
- `Ustart=[]`
- `Sstart=[]`
- `Vstart=[]`

Valori ammessi e vincoli:

- `type`
  Valori supportati: `basic`, `advanced`.
- `function`
  Valori accettati: `empca`, `decomp_empca`, `decomp_CG_means`.
- `function`
  `decomp_empca` e' un alias legacy e viene risolto a `empca`.
- `type='advanced'`
  Richiede obbligatoriamente `function='decomp_CG_means'`.
- `Vimposed.type`
  Valori supportati: `None`, `Heaviside`, `Linear`, `V`.
- `offsets_epoch_imposed`
  E' un alias di compatibilita' per definire `Heaviside` via epoche.
- `offsets_epoch_imposed`
  E' compatibile solo con `Vimposed.type='None'` oppure `Vimposed.type='Heaviside'`.
- `offsets_epoch_imposed`
  Non usare insieme a `Vimposed.param` quando `Vimposed.type='Heaviside'`.
- `Ustart`, `Sstart`, `Vstart`
  Se ne passi uno, devi passare anche gli altri due.

Scelta consigliata:

- `basic + empca` e' il default robusto e il punto di partenza migliore.
- `advanced + decomp_CG_means` ha senso quando vuoi riprodurre il ramo legacy con media/decomposizione congiunta.
- `iter_max`
  Parti da `1e6` e riduci solo se stai facendo scansioni molto ampie.
- `tol`
  `1e-7` e' un compromesso pratico; valori piu' piccoli possono migliorare l'accuratezza ma rallentano molto.
- `Vimposed`
  `Heaviside` ha senso per step co-sismici a epoche note, `Linear` per un trend imposto, `V` quando vuoi passare direttamente una matrice temporale completa.

## Reference Di decompositionPCA

Default effettivi:

- `decomp_fcn='empca'`
- `iter_max_decomp=500000`
- `tol_decomp=1e-7`
- `rand_missingdata=0`
- `rand_init=0`

Valori ammessi e vincoli:

- `decomp_fcn`
  Valori accettati: `empca`, `decomp_empca`, `decomp_srebro_CG_simultaneous`.
- `decomp_fcn`
  `decomp_empca` e' un alias legacy e viene risolto a `empca`.
- `decomp_fcn`
  Implementazioni Python reali: `empca`, `decomp_srebro_CG_simultaneous`.
- `rand_init`
  Deve restare `0`; valori diversi non sono ancora implementati nel port Python.

Scelta consigliata:

- `empca` e' il default consigliato.
- `decomp_srebro_CG_simultaneous` ha senso quando vuoi allinearti al ramo legacy simultaneo e accetti tempi maggiori.
- `tol_decomp`
  Parti da `1e-7`; scendi solo se hai una ragione chiara di accuratezza.

## Reference Di decompositionICA

Default effettivi:

- `source_type='g'`
- `learning_percent=100`
- `ICA_num=1`
- `n_mixed_pdfs=4`
- `states=null`
- `mix_prior_preset=null`
- `source_prior_preset=null`
- `net_init='SVD'`
- `source_init='kmeans'`
- `max_steps=500`
- `isonoise=1`
- `ARD=1`
- `tol=1e-8`
- `eta=1.0`
- `mix.b_alpha_0=1e3`
- `mix.c_alpha_0=1e-3`
- `noise.b_Lam_0=1e1`
- `noise.c_Lam_0=1e-1`
- `noise.mb0=1.0`
- `noise.mn0=0.0`
- `source.m_0=0.0`
- `source.tau_0=1.0`
- `source.b_0=1e1`
- `source.c_0=1e-1`
- `source.lambda_0=null`
- `source.setSource=1`

Valori ammessi e vincoli:

- `source_type`
  Nel port Python oggi e' implementato solo `g`.
- `ICA_num`
  Nel port Python oggi e' implementato solo `1`.
- `source_init`
  Nel port Python oggi e' implementato solo `kmeans`.
- `net_init`
  Valori implementati: `SVD`, `SVD_S&J`.
- `mix_prior_preset`
  Valori supportati: `legacy_r1`, `legacy_r2`, `legacy_r3`, `legacy_r4`.
- `source_prior_preset`
  Valori supportati: `legacy_o1`, `legacy_o2`, `legacy_o3`, `legacy_o4`.
- `states`
  Se lasci `null`, il port usa `n_mixed_pdfs` per ogni componente.
- `n_mixed_pdfs`, `states`, `mix.*`, `source.m_0`, `source.tau_0`, `source.b_0`, `source.c_0`, `source.lambda_0`
  Puoi passare un singolo valore oppure un array di lunghezza `1` o `n_components`.
- `source.lambda_0`
  Se lasci `null`, il port usa una regola legacy dipendente dal numero di campioni del problema vbICA: `n_epochs` in `t-mode`, `n_series` in `s-mode`.

Preset legacy disponibili:

- `source_prior_preset='legacy_o1'`
  Risolve a `b_0=1e3`, `c_0=1e-3`.
- `source_prior_preset='legacy_o2'`
  Risolve a `b_0=1e1`, `c_0=1e-1`.
- `source_prior_preset='legacy_o3'`
  Risolve a `b_0=1e-1`, `c_0=1e1`.
- `source_prior_preset='legacy_o4'`
  Risolve a `b_0=1e-3`, `c_0=1e3`.
- `mix_prior_preset='legacy_r1'`
  Risolve a `b_alpha_0=1e5`, `c_alpha_0=1e-1`.
- `mix_prior_preset='legacy_r2'`
  Risolve a `b_alpha_0=1e1`, `c_alpha_0=1e-1`.
- `mix_prior_preset='legacy_r3'`
  Risolve a `b_alpha_0=1e-1`, `c_alpha_0=1e1`.
- `mix_prior_preset='legacy_r4'`
  Risolve a `b_alpha_0=1e-3`, `c_alpha_0=1e3`.

Scelta consigliata:

- `net_init='SVD'` e' il default piu' stabile.
- `net_init='SVD_S&J'` e' utile come alternativa di ricerca, ma confronta sempre `summary.json`.
- `n_mixed_pdfs=4` e' il primo valore da mantenere per confronti legacy.
- Per `mix.*` e `source.*` conviene esplorare ordini di grandezza come `1e5`, `1e3`, `1e1`, `1e-1`, `1e-3` piu' che micro-variazioni fini.

## Reference Compatta Delle Sezioni Di Supporto

Queste sezioni esistono nella configurazione condivisa e sono visibili nella base completa [default.config.json](../../Scenarios/casestudy/case1/python_port/config/default.config.json). Oggi non guidano il flusso numerico principale di `run_decomposition`, ma sono comunque chiavi valide del modello di configurazione.

- `velocity`
  Chiavi: `file`, `format`.
- `flags`
  Chiavi: `flag_detrend`, `flag_disp`, `flag_decomp_err`, `flag_fault_model`, `flag_load_rand_guess`, `flag_load_rand`, `mixnet_flag`, `flag_visible`, `flag_whitening`, `flag_plot_ts`, `flag_plot_ts_offsets`, `flag_rm_offsets`, `flag_ICA_decomp`, `flag_invert_ICs`, `flag_invert_offsets`.
- `outliers`
  Chiavi: `blunder_threshold_unit`, `blunder_threshold_hor`, `blunder_threshold_ver`, `outlier_threshold`, `centering.*`, `decompositionPCA.*`.
- `outliers.centering`
  Chiavi: `type`, `function`, `n_components`, `n_comp_mean`, `iter_max`, `tol`, `func`, `dfunc`, `Vimposed.*`.
- `outliers.decompositionPCA`
  Chiavi: `iter_max_decomp`, `tol_decomp`, `decomp_fcn`, `rand_missingdata`, `rand_init`.

## Sezioni Documentate Ma Non Ancora Operative Nel Flusso Principale

- `flags`
  Sono accettati e salvati nell'output, ma oggi non cambiano il flusso computazionale Python come fanno nel legacy MATLAB.
- `outliers`
  Esposti per coprire i `parameter_files`, ma il preprocessing outlier dedicato non e' ancora implementato nel port Python.
- `velocity`
  Non entra nel flusso principale di `run_decomposition`, ma resta nella configurazione per compatibilita' con il percorso MATLAB clean/legacy quando si abilita il detrending.

## Cross-Checks Consigliati

- Quando confronti scomposizioni con lo stesso dataset e preprocessing, usa soprattutto `summary.json` e `quality_metrics`.
- Per confronti tra run multiple usa `compare_decomposition_runs.py` sui file `all_python.npz`.
- `resolved_config` e' il riferimento migliore quando vuoi vedere i nomi normalizzati, i preset risolti e i parametri per componente realmente usati.
- Con `decomposition_mode='s'`, `PCA_U/ICA_U` e `PCA_V/ICA_V` vengono rimappate alle shape storiche `(n_series,k)` e `(n_epochs,k)`; l'indipendenza ICA pero' e' stimata nel problema trasposto.
- `ICA_energy` e `ARD_ratio` hanno senso solo tra run comparabili; `variance_explained_*` e `reduced_chi2_*` restano le metriche piu' immediate per il fit.
