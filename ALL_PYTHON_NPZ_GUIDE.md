# Guida Alle Variabili Di `all_python.npz`

`all_python.npz` e' l'archivio NumPy compresso prodotto dalla pipeline Python di case 1.
Contiene:

- matrici osservate e ricostruite
- fattorizzazioni PCA e ICA
- metriche flat per confronti batch
- JSON serializzati con configurazione e metriche strutturate

Questa guida descrive cosa rappresenta ogni variabile, come leggerne shape e indici, e come cambia il layout nei casi `GPS1`, `GPS2`, `GPS3`.

## Notazione

In questa guida:

- `n_stations` = numero di stazioni GPS
- `component_size` = numero di componenti osservate per stazione
- `n_series = n_stations * component_size`
- `n_epochs` = numero di epoche temporali
- `k` = numero di componenti PCA/ICA richieste da `n_components`

## Layout Dei Dati Per `GPS1`, `GPS2`, `GPS3`

Il tipo dataset determina quante serie osservate ha ciascuna stazione e in che ordine vengono salvate nelle righe di tutte le matrici con shape `(n_series, ...)`.

| Dataset type | Componenti per stazione | `component_size` | Ordine righe per stazione |
| --- | --- | ---: | --- |
| `GPS1` | solo verticale | `1` | `u` |
| `GPS2` | piano orizzontale | `2` | `e`, `n` |
| `GPS3` | 3D completo | `3` | `e`, `n`, `u` |

Quindi:

- in `GPS1`, la riga `s` corrisponde alla componente `u` della stazione `s`
- in `GPS2`, le righe `2*s` e `2*s+1` corrispondono a `e` e `n`
- in `GPS3`, le righe `3*s`, `3*s+1`, `3*s+2` corrispondono a `e`, `n`, `u`

Esempi:

- `Xd_name`: `ANCGe`, `ANCGn`, `ANCGu`, `ANCNe`, `ANCNn`, `ANCNu`, ...
- `Xd_type`: `GPS3e`, `GPS3n`, `GPS3u`, `GPS3e`, `GPS3n`, `GPS3u`, ...
- `STATIONS_name`: `ANCG`, `ANCN`, `ARCE`, ...

La stessa convenzione di righe vale per:

- `Xd_ts`, `Xd_var_ts`, `Xd_llh`
- `PCA_U`, `PCA_ts`
- `ICA_U`, `ICA_ts`
- `A_recon`, `var_A_recon`
- `data_mask`

## Come Leggere `ICA_U`, `ICA_S`, `ICA_V`

Questa e' la parte piu' importante per evitare equivoci.

Le colonne non rappresentano `East`, `North`, `Up`.
Le colonne rappresentano le componenti ICA.

Per ogni componente `j` con `0 <= j < k`:

- `ICA_U[:, j]` e' il pattern spaziale della componente `j` su tutte le serie osservate
- `ICA_S[j, j]` e' il fattore di scala della componente `j`
- `ICA_V[:, j]` e' la serie temporale della componente `j`

La ricostruzione della sola componente `j` e':

```text
ICA_component_j = ICA_U[:, j] * ICA_S[j, j] * ICA_V[:, j]^T
```

La ricostruzione completa e':

```text
ICA_ts = ICA_U @ ICA_S @ ICA_V.T
```

Interpretazione corretta:

- la colonna `j=0` non e' "Est"
- la colonna `j=1` non e' "Nord"
- la colonna `j=2` non e' "Up"
- `e`, `n`, `u` stanno nelle righe di `ICA_U`, non nelle colonne

Quindi, per isolare il contributo spaziale della componente ICA `j`:

- `GPS1`: `up = ICA_U[:, j]`
- `GPS2`: `east = ICA_U[0::2, j]`, `north = ICA_U[1::2, j]`
- `GPS3`: `east = ICA_U[0::3, j]`, `north = ICA_U[1::3, j]`, `up = ICA_U[2::3, j]`

## Ordinamento E Segno Delle Componenti

Le componenti ICA e PCA vengono ordinate in ordine decrescente rispetto ai valori diagonali di `S`.
Quindi:

- la prima colonna e' la componente con scala piu' grande in `S`
- non e' una direzione fisica
- il segno di `U` e `V` puo' essere ribaltato insieme senza cambiare la ricostruzione

Per ICA, il file salva una versione normalizzata:

- `ICA_U` deriva da `A_recon` normalizzata colonna per colonna
- `ICA_V` deriva da `S_recon` normalizzata riga per riga
- `ICA_S` raccoglie le norme spostate fuori da `A_recon` e `S_recon`

Quindi:

- `A_recon`, `S_recon` = fattori grezzi della rete VBICA
- `ICA_U`, `ICA_S`, `ICA_V` = fattorizzazione normalizzata e ordinata utile per analisi e plotting

## Variabili Principali

### Dati osservati `Xd_*`

| Variabile | Shape | Significato | Note |
| --- | --- | --- | --- |
| `Xd_ts` | `(n_series, n_epochs)` | Serie osservate dopo i filtri e la centratura | Ogni riga e' una componente osservata di una stazione |
| `Xd_var_ts` | `(n_series, n_epochs)` | Varianza associata a `Xd_ts` | I missing sono rappresentati con `inf` |
| `Xd_timeline` | `(n_epochs,)` | Asse temporale comune | Stesse epoche usate da PCA e ICA |
| `Xd_llh` | `(n_series, 3)` | Longitudine, latitudine, quota per ogni riga di serie | Le coordinate sono ripetute per `e/n/u` della stessa stazione |
| `Xd_centering_offsets` | `(n_series,)` | Offset rimossi in fase di centratura | Da riaggiungere per tornare al livello originale |
| `Xd_name` | `(n_series,)` | Nome completo della serie | Esempio `ANCGe`, `ANCNn`, `ARCGu` |
| `Xd_type` | `(n_series,)` | Tipo completo della serie | Esempio `GPS3e`, `GPS2n`, `GPS1u` |

### PCA

| Variabile | Shape | Significato | Note |
| --- | --- | --- | --- |
| `PCA_U` | `(n_series, k)` | Pattern spaziali PCA | Righe ordinate come `GPS1/2/3` |
| `PCA_S` | `(k, k)` | Matrice diagonale delle scale PCA | Diagonale ordinata in ordine decrescente |
| `PCA_V` | `(n_epochs, k)` | Pattern temporali PCA | La colonna `j` e' la serie temporale della componente `j` |
| `PCA_ts` | `(n_series, n_epochs)` | Ricostruzione PCA nel dominio dei dati | `PCA_U @ PCA_S @ PCA_V.T` |
| `PCA_decomposition_mode` | scalare stringa | Modalita' usata dalla decomposizione PCA | `t` oppure `s`; in `s` i fattori sono rimappati alle shape storiche |

### ICA normalizzata

| Variabile | Shape | Significato | Note |
| --- | --- | --- | --- |
| `ICA_U` | `(n_series, k)` | Pattern spaziali ICA | Le colonne sono componenti ICA, non direzioni `e/n/u` |
| `ICA_S` | `(k, k)` | Matrice diagonale delle scale ICA | Le componenti sono ordinate per diagonale decrescente |
| `ICA_V` | `(n_epochs, k)` | Pattern temporali ICA | La colonna `j` e' la storia temporale della componente `j` |
| `ICA_ts` | `(n_series, n_epochs)` | Ricostruzione ICA nel dominio dei dati | `ICA_U @ ICA_S @ ICA_V.T` |
| `ICA_llh` | `(n_series, 3)` | Coordinate associate alla soluzione ICA | Normalmente uguali a `Xd_llh` |
| `ICA_timeline` | `(n_epochs,)` | Asse temporale associato alla soluzione ICA | Normalmente uguale a `Xd_timeline` |
| `ICA_name` | `(n_series,)` | Nomi serie associati a ICA | Normalmente uguali a `Xd_name` |
| `ICA_type` | `(n_series,)` | Tipi serie associati a ICA | Normalmente uguali a `Xd_type` |
| `ICA_decomposition_mode` | scalare stringa | Modalita' usata dalla decomposizione ICA | `t` oppure `s`; in `s` la vbICA e' stimata sulla matrice trasposta |

### Fattori grezzi VBICA

| Variabile | Shape | Significato | Note |
| --- | --- | --- | --- |
| `A_recon` | `(n_series, k)` | Mixing matrix grezza della rete ICA | Prima della normalizzazione che porta a `ICA_U` |
| `S_recon` | `(k, n_epochs)` | Sorgenti grezze ricostruite dalla rete ICA | Prima della normalizzazione che porta a `ICA_V` |
| `var_A_recon` | `(n_series, k)` | Varianza posteriore di `A_recon` | Derivata dalle precisioni posteriori |
| `var_S_recon` | `(k, n_epochs)` | Varianza posteriore di `S_recon` | Derivata dalle precisioni posteriori |

Nota: in `decomposition_mode='s'`, questi fattori grezzi sono la trasposizione coerente del problema vbICA interno; mantengono le shape storiche per compatibilita' con analisi, metriche e grafici.

### Missing data e metadati stazione

| Variabile | Shape | Significato | Note |
| --- | --- | --- | --- |
| `data_mask` | `(n_series, n_epochs)` | Maschera dati validi | `1` = dato presente, `0` = dato mancante |
| `ind_missing_data` | `(n_missing,)` | Indici flat dei missing | Indici 1-based, mantenuti cosi' per compatibilita' con MATLAB |
| `STATIONS_name` | `(n_stations,)` | Codici stazione senza suffisso componente | Un elemento per stazione, non per serie |

## Metriche Flat

Queste chiavi sono pensate per confronto rapido tra run e vengono anche riusate dagli script batch.

| Variabile | Shape | Significato | Note |
| --- | --- | --- | --- |
| `chi2_PCA` | scalare | Chi-quadro totale della ricostruzione PCA | |
| `chi2_ICA` | scalare | Chi-quadro totale della ricostruzione ICA | |
| `variance_explained_PCA` | scalare | Percentuale di varianza spiegata dalla PCA | In percento |
| `variance_explained_ICA` | scalare | Percentuale di varianza spiegata dalla ICA | In percento |
| `ard` | `(k,)` | Pesi ARD normalizzati | Derivati dagli `alpha` ICA |
| `n_observations` | scalare intero | Numero di osservazioni valide | Esclude i missing |
| `missing_data_fraction` | scalare | Frazione di dati mancanti | Calcolata da `Xd_var_ts` |
| `reduced_chi2_PCA` | scalare | Chi-quadro ridotto PCA | |
| `reduced_chi2_ICA` | scalare | Chi-quadro ridotto ICA | |
| `weighted_rms_PCA` | scalare | RMS pesato PCA | |
| `weighted_rms_ICA` | scalare | RMS pesato ICA | |
| `chi2_gain_ICA_vs_PCA_pct` | scalare | Guadagno percentuale di ICA rispetto a PCA in termini di chi-quadro | Maggiore e' meglio |
| `var_explained_gain_ICA_vs_PCA` | scalare | Guadagno di varianza spiegata di ICA rispetto a PCA | |
| `ICA_energy` | scalare | Energia finale della rete ICA | Di solito coincide con `ICA_net_energy` |
| `ICA_iterations` | scalare intero | Numero iterazioni ICA | Di solito uguale a `len(ICA_net_energy_path)` |
| `ARD_ratio` | scalare | `max(alpha) / min(alpha)` | Valori alti suggeriscono componenti in eccesso |

## Diagnostica Di Fit E Della Rete ICA

| Variabile | Shape | Significato | Note |
| --- | --- | --- | --- |
| `PCA_fit_method` | scalare stringa | Metodo usato per la PCA | Esempio `empca`, `srebro` |
| `PCA_fit_iterations` | scalare intero | Numero iterazioni del fit PCA | |
| `PCA_fit_objective_name` | scalare stringa | Nome dell'obiettivo ottimizzato | Esempio `chi2`, `residual` |
| `PCA_fit_objective_final` | scalare | Valore finale dell'obiettivo | |
| `PCA_fit_objective_path` | `(n_iter+1,)` oppure assente | Traccia completa dell'obiettivo PCA | Presente per `empca` |
| `ICA_net_energy` | scalare | Energia finale della rete ICA | Duplicato utile per accesso diretto |
| `ICA_net_energy_path` | `(n_iter,)` | Evoluzione dell'energia ICA durante il training | Utile per vedere convergenza |
| `ICA_net_alphas` | `(k,)` | Iperparametri ARD grezzi della rete ICA | Da questi deriva `ard` |

## JSON Serializzati

Queste chiavi contengono stringhe JSON, non array numerici.

| Variabile | Tipo | Significato | Note |
| --- | --- | --- | --- |
| `config_json` | stringa JSON | Configurazione iniziale della run | Contiene i parametri principali richiesti all'avvio |
| `resolved_config_json` | stringa JSON | Configurazione effettiva risolta | Include anche campi risolti dal codice, per esempio `seismicity_resolved` |
| `quality_metrics_json` | stringa JSON | Metriche strutturate complete | Contiene i blocchi `data`, `PCA`, `ICA` |

## Cosa Cambia Tra `GPS1`, `GPS2`, `GPS3`

Le chiavi del file restano le stesse.
Cambiano:

- `component_size`
- `n_series`
- l'ordine delle righe nelle matrici spaziali
- il modo in cui devi leggere una colonna di `U`

### Caso `GPS1`

- `component_size = 1`
- c'e' solo la componente `u`
- `n_series = n_stations`
- una colonna `ICA_U[:, j]` ha un valore per stazione, gia' interpretabile come contributo verticale

### Caso `GPS2`

- `component_size = 2`
- l'ordine per stazione e' `e`, `n`
- `n_series = 2 * n_stations`
- una colonna `ICA_U[:, j]` va separata in:

```text
east  = ICA_U[0::2, j]
north = ICA_U[1::2, j]
```

### Caso `GPS3`

- `component_size = 3`
- l'ordine per stazione e' `e`, `n`, `u`
- `n_series = 3 * n_stations`
- una colonna `ICA_U[:, j]` va separata in:

```text
east  = ICA_U[0::3, j]
north = ICA_U[1::3, j]
up    = ICA_U[2::3, j]
```

Lo stesso schema vale per `PCA_U`, `A_recon`, `var_A_recon`, `Xd_ts`, `ICA_ts`, `data_mask`, e per ogni altra matrice indicizzata per serie.

## Chiavi Sempre Presenti E Chiavi Condizionali

Normalmente sono sempre presenti:

- tutti i campi `Xd_*`
- tutti i campi `PCA_*` e `ICA_*` elencati sopra
- `A_recon`, `S_recon`, `var_A_recon`, `var_S_recon`
- `data_mask`, `ind_missing_data`, `STATIONS_name`
- le metriche flat principali
- `config_json`, `resolved_config_json`, `quality_metrics_json`

Possono essere condizionali:

- `PCA_fit_objective_path`: presente per alcuni metodi PCA, in particolare `empca`
- `ICA_net_energy`, `ICA_net_energy_path`, `ICA_net_alphas`: presenti se la soluzione ICA salva la diagnostica completa della rete

## Mappa Rapida Da Tenere A Mente

- colonne di `U` = componenti
- colonne di `V` = componenti
- righe di `U` = serie osservate
- righe di `Xd_ts` e `ICA_ts` = serie osservate
- `e/n/u` dipendono dal dataset type e stanno nelle righe
- `GPS1 -> u`
- `GPS2 -> e, n`
- `GPS3 -> e, n, u`
- `A_recon` e `S_recon` sono i fattori grezzi
- `ICA_U`, `ICA_S`, `ICA_V` sono la forma normalizzata e ordinata
