# PageRank Feasibility Audit of Wikipedia RfA Support Votes

## Project Overview

This repository contains the final mini-project for **ADAN7905 — AI / ML Software Tools and Platforms**. 
Primary Dataset: Stanford Network Analysis Project (SNAP) Wikipedia Requests for Adminship (RfA). 
Purpose: Study PageRank behavior in a real directed social-voting network.

The project builds a directed support-vote graph, computes PageRank, labels each support edge as PageRank-uphill or non-uphill, engineers graph-based predictors, trains supervised machine-learning models, produces exploratory and model-diagnostic visualizations, and incorporates support, oppose, and neutral votes through a lifted signed PageRank component.

The project is designed to satisfy the full mini-project workflow:

```text
raw data
→ data cleaning
→ data wrangling
→ graph construction
→ PageRank computation
→ feature engineering
→ data exploration
→ train/test split
→ machine-learning classification
→ model diagnostics
→ lifted signed PageRank
→ results and conclusions
→ Docker/GitHub reproducibility
```

## Research Question

**Do Wikipedia RfA support-vote edges tend to point toward users with higher PageRank, and can this PageRank-uphill behavior be predicted from local graph features?**

In this project, each directed support edge is interpreted as:

```text
source voter → target candidate
```

An edge is labeled **PageRank-uphill** if:

```text
PageRank(target) > PageRank(source)
```

This converts the graph-theoretic PageRank question into an edge-level supervised classification problem.

The project also asks a second, related signed-network question:

**How does the centrality interpretation change when support, oppose, and neutral votes are incorporated into the analysis rather than using support votes alone?**

## Analytical Objective

The project asks whether real support-vote behavior in Wikipedia RfA is consistent with a strict one-sided PageRank domination condition. If a domination condition required source nodes to dominate target nodes along support edges, then many edges with `PageRank(target) > PageRank(source)` would represent empirical domination violations.

The analysis will use PageRank as:
1. A **network centrality measure**, identifying structurally central users in the support-vote graph.
2. A **diagnostic framework**, measuring whether support edges usually flow toward or away from higher-PageRank users.
3. A **signed-network centrality framework**, distinguishing positive support, negative opposition, neutral attention, total signed attention, and net signed reputation.

## Course / Rubric Alignment

| Requirement | How this repository satisfies it |
|---|---|
| GitHub deliverable | The repository contains a Dockerfile, requirements.txt, code, raw data, processed/final data, figures, outputs, README.md, and Report.pdf. |
| Clear research question | The project asks whether support-vote edges point uphill in PageRank, whether that behavior is predictable, and how signed vote information changes the centrality interpretation. |
| High-quality dataset | The project uses the SNAP Wikipedia Requests for Adminship dataset, a legitimate public research dataset containing directed signed voting records. |
| Data wrangling | The raw text data are parsed into structured vote and edge tables. |
| Data cleaning | The pipeline removes malformed records, blank source/target values, self-loops, and organizes votes into support, oppose, and neutral layers. |
| Data split | The modeling script uses a stratified train/test split for supervised classification. |
| Feature engineering | The project creates edge-level graph features from source/target degree structure, degree differences, reciprocity, and support count. |
| Data exploration | The project provides graph summaries, PageRank summaries, degree distributions, PageRank-difference distributions, uphill-edge counts, signed centrality summaries, rank-shift diagnostics, correlation tables, and model-diagnostic figures. |
| Modeling and prediction | The project trains a majority-class baseline, logistic regression, and random forest classifier to predict PageRank-uphill edges. |
| Results and conclusions | The report provides a scientific interpretation of results, limitations, next steps, and the role of the tech stack. |
| Visualization | The repository includes multiple EDA, model-diagnostic, signed-network, and interpretive figures. |
| Docker reproducibility | The full pipeline runs successfully inside Docker. |
| README / Report | This README documents the complete project, and Report.pdf provides a formatted written report if included. |

## Dataset

The raw data file is:

```text
data/raw/wiki-RfA.txt
```

The dataset is the **SNAP Wikipedia Requests for Adminship (with text)** dataset. It is a directed, signed voting network in which nodes represent Wikipedia users and edges represent votes in the RfA process.

The raw fields include:

| Field | Meaning |
|---|---|
| `SRC` | Source user / voter |
| `TGT` | Target user / candidate |
| `VOT` | Vote sign: support, neutral, or oppose |
| `RES` | Result of the RfA nomination |
| `YEA` | Year |
| `DAT` | Date/time |
| `TXT` | Vote comment text |

Dataset source:

```text
https://snap.stanford.edu/data/wiki-RfA.html
```

### Dataset Strengths

The dataset is well-suited for this project because it is:

- A real-world public research dataset.
- Directed, which is required for PageRank.
- Signed, allowing support, neutral, and oppose votes to be distinguished.
- Large enough to support meaningful graph and machine-learning analysis.
- Socially interpretable because edges represent actual voting behavior.
- Compatible with graph analytics, feature engineering, supervised learning, signed-network analysis, and visualization.

### Dataset Weaknesses

Limitations of the dataset include:

- It is historical and covers RfA activity from 2003 through May 2013.
- Vote text is retained in the cleaned table but not used in the main model.
- PageRank results depend on graph construction choices, including filtering, duplicate aggregation, damping factor, and whether the graph is treated as unsigned or signed.
- The model is based on observed graph structure, not causal social mechanisms.
- Wikipedia usernames may contain spaces, punctuation, non-English characters, or URL-encoded strings. These values were retained as valid user identifiers unless the source or target field was blank or missing.


## Repository Structure

```text
pagerank-rfa-feasibility-audit/
|
├── Dockerfile
├── README.md
├── Report.pdf
├── requirements.txt
├── .gitignore
|
├── data/
|   ├── raw/
|   |   └── wiki-RfA.txt
|   └── processed/
|       ├── rfa_votes_clean.csv
|       ├── rfa_support_edges.csv
|       └── edge_features.csv
|
├── src/
|   ├── 02_clean_data.py
|   ├── 03_build_graph.py
|   ├── 04_compute_features.py
|   ├── 05_make_figures.py
|   ├── 06_model_edges.py
|   ├── 07_advanced_figures.py
|   ├── 08_signed_pagerank_extension.py
|   └── run_pipeline.py
|
├── figures/
|   ├── confusion_matrix.png
|   ├── neutral_vs_support_pagerank.png
|   ├── pagerank_distribution.png
|   ├── pagerank_edge_difference.png
|   ├── precision_recall_curve_random_forest.png
|   ├── random_forest_feature_importance.png
|   ├── roc_curve_random_forest.png
|   ├── signed_net_score_distribution.png
|   ├── signed_rank_shift_top20.png
|   ├── signed_support_vs_oppose.png
|   ├── source_in_degree_distribution.png
|   ├── target_in_degree_distribution.png
|   ├── uphill_edge_counts.png
|   └── uphill_share_by_year.png
|
└── outputs/
    ├── graph_summary.json
    ├── model_metrics.csv
    ├── pagerank_scores.csv
    ├── signed_pagerank_scores.csv
    ├── signed_pagerank_top_nodes.csv
    ├── signed_score_correlations.csv
    └── top_25_pagerank_nodes.csv
```

## Script Overview

| Script | Purpose |
|---|---|
| `src/02_clean_data.py` | Parses raw SNAP text file, cleans vote records, removes unusable rows, filters to support votes for the baseline graph, and creates the support edge list. |
| `src/03_build_graph.py` | Builds directed support graph with NetworkX, computes support-only PageRank, and saves graph summaries. |
| `src/04_compute_features.py` | Creates edge-level features and labels each support edge as PageRank-uphill or non-uphill. |
| `src/05_make_figures.py` | Generates standard exploratory figures before modeling. |
| `src/06_model_edges.py` | Trains and evaluates baseline, logistic regression, and random forest models; saves model metrics and confusion matrix. |
| `src/07_advanced_figures.py` | Generates model-diagnostic and temporal figures, including ROC curve, precision-recall curve, feature importance, and uphill share by year. |
| `src/08_signed_pagerank_extension.py` | Computes lifted signed PageRank using support and oppose votes, computes neutral PageRank as an attention layer, and saves signed-network outputs and figures. |
| `src/run_pipeline.py` | Runs full project pipeline in order. |

## Methods

### 1. Data Cleaning

The script `src/02_clean_data.py` parses the raw SNAP text file and creates two processed files:

```text
data/processed/rfa_votes_clean.csv
data/processed/rfa_support_edges.csv
```

Cleaning steps include:

1. Parsing labeled raw text records.
2. Standardizing columns as `source`, `target`, `vote`, `result`, `year`, `date`, and `text`.
3. Dropping malformed records.
4. Dropping missing or blank source/target values.
5. Removing self-loops.
6. Filtering to support votes (`vote == 1`) for the main support-only PageRank graph.
7. Aggregating duplicate source-target support pairs into a unique directed edge list with `support_count`.
8. Retaining the full cleaned vote table so support, oppose, and neutral votes can be used in the signed PageRank component.

### 2. Graph Construction

The script `src/03_build_graph.py` loads the support edge list into a NetworkX directed graph.

In the support-only graph:

```text
node = Wikipedia user
directed edge = support vote from source voter to target candidate
```

The final support graph is a directed support-vote network where an edge from user `i` to user `j` means that user `i` supported user `j` in an RfA vote.

### 3. PageRank Computation

PageRank is computed using:

```text
alpha = 0.85
```

The damping factor means that the random walk follows graph edges 85% of the time and teleports 15% of the time. In this analysis, support-only PageRank is interpreted as a structural centrality score in the support-vote network. A higher PageRank user is a user who is more central as a recipient of support from other structurally important users.

### 4. PageRank-Uphill Labeling

Each directed support edge is labeled as:

```text
uphill = 1 if PageRank(target) > PageRank(source)
uphill = 0 otherwise
```

This label identifies whether the support edge points toward a higher-PageRank user.

### 5. Feature Engineering

The feature table includes the following predictors:

| Feature | Meaning |
|---|---|
| `support_count` | Number of support votes observed for a source-target pair after aggregation. |
| `source_in_degree` | Number of support edges received by the source voter. |
| `source_out_degree` | Number of support edges sent by the source voter. |
| `target_in_degree` | Number of support edges received by the target candidate. |
| `target_out_degree` | Number of support edges sent by the target candidate. |
| `in_degree_diff` | `target_in_degree - source_in_degree`. |
| `out_degree_diff` | `target_out_degree - source_out_degree`. |
| `reciprocal` | Indicator for whether the target also has an edge back to the source. |

The project intentionally excludes PageRank itself, source PageRank, target PageRank, and PageRank difference from the predictive feature set to avoid direct target leakage.


### 6. Data Exploration

Exploratory data analysis is performed after cleaning, graph construction, and feature engineering, and before supervised modeling. The EDA stage examines:

- Cleaned vote counts by vote sign.
- Support-only edge count.
- Graph size, density, and component structure.
- Top support-only PageRank users.
- PageRank distribution.
- Source and target in-degree distributions.
- Edge-level PageRank differences.
- Class balance between PageRank-uphill and non-uphill edges.
- Temporal variation in PageRank-uphill share by year.

The main EDA figures are:

```text
pagerank_distribution.png
pagerank_edge_difference.png
source_in_degree_distribution.png
target_in_degree_distribution.png
uphill_edge_counts.png
uphill_share_by_year.png
```

These summaries are used to understand the model-ready graph data before evaluating supervised classifiers.

### 7. Train/Test Split

The model uses a stratified train/test split:

```text
test_size = 0.20
random_state = 42
stratify = uphill
```

A stratified split is appropriate because the target class is imbalanced: PageRank-uphill edges are substantially more common than non-uphill edges. The project compares fixed models rather than tuning hyperparameters extensively, so a separate validation set was not necessary for this project scope. A future extension could add cross-validation or a temporal train/validation/test split.

### 8. Modeling

The script `src/06_model_edges.py` trains three models:

1. **Majority-class baseline**
2. **Logistic regression**
3. **Random forest classifier**

The majority-class baseline provides a reference point. Logistic regression provides an interpretable linear model. The random forest provides a stronger nonlinear classifier capable of capturing interactions among graph features.

Evaluation metrics include:

- Accuracy.
- Precision.
- Recall.
- F1 score.
- ROC-AUC.
- Confusion matrix.

### 9. Lifted Signed PageRank

The script `src/08_signed_pagerank_extension.py` incorporates support, oppose, and neutral votes into a signed-network analysis. Ordinary PageRank cannot directly treat negative edges as negative probabilities, so the signed component uses a lifted state space.

Each user is represented by two signed states:

```text
(i, +) = positive reputation state
(i, -) = negative reputation state
```

Support edges preserve sign:

```text
(i, +) → (j, +)
(i, -) → (j, -)
```

Oppose edges flip sign:

```text
(i, +) → (j, -)
(i, -) → (j, +)
```

This creates a nonnegative Markov chain on a doubled state space, allowing PageRank-style computation while preserving the distinction between positive and negative reputation flow. Neutral votes are modeled separately as a neutral-attention PageRank layer because they do not have a clear positive or negative polarity.

The signed component produces:

| Score | Interpretation |
|---|---|
| `pagerank_support` | Ordinary PageRank on the support layer over the all-vote user universe. |
| `pagerank_oppose` | Ordinary PageRank on the oppose layer. |
| `pagerank_neutral` | Ordinary PageRank on the neutral layer. |
| `signed_positive` | Positive reputation mass from the lifted signed PageRank model. |
| `signed_negative` | Negative reputation mass from the lifted signed PageRank model. |
| `signed_attention` | Total signed mass: `signed_positive + signed_negative`. |
| `signed_net` | Net signed reputation: `signed_positive - signed_negative`. |
| `signed_balance` | Normalized signed balance: `signed_net / signed_attention`. |
| `rank_shift_signed_minus_support` | Rank shift from support-only PageRank to signed-net PageRank. |

The support PageRank values in `signed_pagerank_scores.csv` are computed over the full all-vote user universe so that support, oppose, and neutral layer scores are comparable. They may therefore differ slightly from the baseline support-only PageRank scores in `pagerank_scores.csv`.

## Key Results


### Cleaned Data Summary

| Item | Value |
|---|---:|
| Clean vote records | 196,511 |
| Support-only edges | 139,272 |
| Support votes | 143,822 |
| Oppose votes | 40,637 |
| Neutral votes | 12,052 |

### Graph Summary

| Metric | Value |
|---|---:|
| Nodes | 10,004 |
| Edges | 139,272 |
| Density | 0.001392 |
| Self-loops | 0 |
| Weakly connected components | 23 |
| Strongly connected components | 7,778 |

### Top Support-Only PageRank Nodes

| Rank | Node | PageRank |
|---:|---|---:|
| 1 | Everyking | 0.003764 |
| 2 | Legoktm | 0.003655 |
| 3 | SarahStierch | 0.003604 |
| 4 | HJ Mitchell | 0.003343 |
| 5 | Mkdw | 0.003285 |
| 6 | TenPoundHammer | 0.003160 |
| 7 | Ironholds | 0.002747 |
| 8 | Tokyogirl79 | 0.002733 |
| 9 | Drmies | 0.002708 |
| 10 | Werdna | 0.002621 |

### PageRank-Uphill Edge Summary

| Edge type | Count | Share |
|---|---:|---:|
| PageRank-uphill | 103,065 | 74.00% |
| Non-uphill | 36,207 | 26.00% |

### Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority-class baseline | 0.7400 | 0.7400 | 1.0000 | 0.8506 | 0.5000 |
| Logistic regression | 0.9207 | 0.9345 | 0.9601 | 0.9471 | 0.9705 |
| Random forest | 0.9251 | 0.9468 | 0.9524 | 0.9496 | 0.9783 |

The random forest was the strongest model by F1 score and ROC-AUC.

### Signed PageRank Summary

| Signed PageRank item | Value |
|---|---:|
| Users across all vote layers | 11,377 |
| Missing values in signed score table | 0 |
| Signed attention sum | 1.000000000000 |
| Minimum signed balance | -0.4573 |
| Maximum signed balance | 1.0000 |
| Signed PageRank convergence | 133 iterations |
| Pairwise signed-score correlations | 90 rows |

The signed score table contains the full user universe across support, oppose, and neutral vote layers. The signed-attention distribution sums to approximately 1.0, confirming that the lifted signed PageRank computation produced a valid normalized distribution.

### Top Signed-Net PageRank Nodes

| Rank | Node | Signed net | Signed positive | Signed negative | Support PageRank | Oppose PageRank |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Legoktm | 0.001030 | 0.002125 | 0.001095 | 0.003540 | 0.000052 |
| 2 | SarahStierch | 0.000978 | 0.001717 | 0.000738 | 0.003491 | 0.000049 |
| 3 | HJ Mitchell | 0.000926 | 0.001827 | 0.000901 | 0.003240 | 0.001297 |
| 4 | Can't sleep, clown will eat me | 0.000891 | 0.001379 | 0.000488 | 0.002094 | 0.000381 |
| 5 | Drmies | 0.000890 | 0.001448 | 0.000558 | 0.002624 | 0.000092 |
| 6 | Mkdw | 0.000837 | 0.001604 | 0.000768 | 0.003178 | 0.000576 |
| 7 | Mark Arsten | 0.000762 | 0.001300 | 0.000538 | 0.002290 | 0.000049 |
| 8 | Dabomb87 | 0.000756 | 0.001143 | 0.000388 | 0.002079 | 0.000049 |
| 9 | Werdna | 0.000711 | 0.001462 | 0.000750 | 0.002546 | 0.001164 |
| 10 | Elonka | 0.000702 | 0.001630 | 0.000928 | 0.002181 | 0.002585 |

### Largest Absolute Rank-Shift Examples

The rank-shift table identifies users whose centrality changes most when moving from support-only PageRank to signed-net PageRank.

| Node | Support rank | Signed-net rank | Rank shift |
|---|---:|---:|---:|
| My76Strat | 125 | 11297 | 11172 |
| Carrite | 196 | 11354 | 11158 |
| Kumioko | 346 | 11350 | 11004 |
| Ottava Rima | 439 | 11376 | 10937 |
| Sam Spade | 501 | 11353 | 10852 |

## Visualizations

The repository includes standard exploratory figures, model-diagnostic figures, signed-network figures, and temporal figures.

| Figure | Description |
|---|---|
| `pagerank_distribution.png` | Shows the right-skewed distribution of support-only PageRank scores across users. |
| `pagerank_edge_difference.png` | Shows the distribution of `PageRank(target) - PageRank(source)` across support edges. |
| `source_in_degree_distribution.png` | Shows the source-user in-degree distribution at the edge level. |
| `target_in_degree_distribution.png` | Shows the target-user in-degree distribution at the edge level. |
| `uphill_edge_counts.png` | Shows the count of PageRank-uphill and non-uphill support edges. |
| `confusion_matrix.png` | Shows random forest classification performance on the test set, with integer-formatted cell counts. |
| `roc_curve_random_forest.png` | Shows the random forest model’s ability to separate uphill from non-uphill edges across classification thresholds. |
| `precision_recall_curve_random_forest.png` | Evaluates random forest performance under class imbalance. |
| `random_forest_feature_importance.png` | Identifies which engineered graph features most strongly contribute to predicting PageRank-uphill edges. |
| `uphill_share_by_year.png` | Shows how the share of PageRank-uphill support edges varies by the first year in which each support edge appears. |
| `signed_support_vs_oppose.png` | Compares support-layer PageRank and oppose-layer PageRank. |
| `signed_net_score_distribution.png` | Shows the distribution of net signed reputation scores. |
| `signed_rank_shift_top20.png` | Shows the largest rank shifts between support-only PageRank and signed-net PageRank. |
| `neutral_vs_support_pagerank.png` | Compares neutral-attention PageRank and support-layer PageRank. |

## Results and Conclusions

The cleaned support-vote graph contains 10,004 nodes and 139,272 directed support edges. The raw vote table retained 196,511 usable vote records after cleaning, including 143,822 support votes, 40,637 oppose votes, and 12,052 neutral votes. The final support graph is sparse, with a density of 0.00139, and has 23 weakly connected components and 7,778 strongly connected components. This structure is consistent with a large social voting network: many users participate locally or briefly, while a smaller set of candidates and active voters occupy more central positions.

The central empirical finding is that PageRank-uphill support edges are common. Of the 139,272 support edges, 103,065 edges, or approximately 74.00%, point from a lower-PageRank source to a higher-PageRank target. Only 36,207 edges, or approximately 26.00%, are non-uphill. This means that support votes in the Wikipedia RfA network usually flow toward users who are more structurally central in the support graph. The PageRank distribution is highly right-skewed: most users have very small PageRank scores, while a small number of users receive substantially larger scores. The top support-only PageRank users include Everyking, Legoktm, SarahStierch, HJ Mitchell, and Mkdw, which supports the interpretation that PageRank captures concentrated structural visibility in the support-vote system.

The exploratory figures reinforce this pattern. The target in-degree distribution is shifted toward higher values than the source in-degree distribution, indicating that candidates receiving support are often more connected than voters casting support. The edge PageRank-difference histogram is centered near zero but has a visibly positive mass, while the uphill versus non-uphill bar chart shows a clear majority of uphill support edges. Together, these plots make the domination-violation pattern visible rather than relying only on summary statistics.

The machine-learning results show that uphill behavior is strongly predictable from graph-based features. Because the target is imbalanced toward uphill edges, the majority-class baseline already has an F1 score of 0.8506 but an ROC-AUC of only 0.5000. Logistic regression improves substantially, reaching accuracy of 0.9207, F1 of 0.9471, and ROC-AUC of 0.9705. The random forest performs best, with accuracy of 0.9251, precision of 0.9468, recall of 0.9524, F1 of 0.9496, and ROC-AUC of 0.9783. The ROC curve and precision-recall curve further show that the random forest separates uphill from non-uphill edges well across classification thresholds, while the feature-importance plot shows which graph features contribute most strongly to prediction.

The signed PageRank component provides a more holistic view of RfA centrality by incorporating support, oppose, and neutral votes. The lifted signed PageRank model converged in 133 iterations and produced a valid signed-attention distribution summing to approximately 1.0 with no missing score values. The signed-net ranking overlaps with the support-only ranking but changes the interpretation for users who receive substantial opposition. For example, Everyking is ranked first by support-only PageRank but lower by signed-net PageRank, while Legoktm rises to the top signed-net position. This shows that the signed component captures additional reputational structure from opposition and neutral attention.

Results support the project’s PageRank feasibility interpretation. A strict one-sided domination condition requiring source nodes to dominate target nodes along support edges would not describe this real network well. Instead, the empirical pattern is the reverse for most edges in that support frequently moves from less central voters toward more central candidates. The signed-network component shows that centrality in the RfA network is multidimensional. Once opposition and neutral votes are incorporated, the analysis distinguishes support centrality, opposition centrality, neutral attention, total signed attention, and net signed reputation.

## Limitations and Next Steps

The project is best interpreted as an empirical audit of an observed historical network, not a causal model of Wikipedia governance. The dataset provided by SNAP were RfA votes from 2003 through May 2013. Therefore, we can not assumed the data, analysis, nor the predictions are equally valid representations of behavior on Wikipedia today. The results may also be limited based on graph-construction choices. For example, the choices whether to aggregate the many duplicate support pairs that are present in the network, and whether to use a weighted or unweighted version of PageRank.

The classifier is an edge-level model on an observed graph. A stricter future-prediction design would compute PageRank and graph features on a training graph only, then evaluate on temporally held-out edges. Reasonable next steps could include: adding PageRank damping-factor sensitivity checks, comparing weighted and unweighted PageRank, adding text-derived vote-comment features, modeling election outcomes directly, or using a temporal split where early RfA behavior predicts later edge patterns. Future work on signed-networks would be to compare the lifted signed PageRank model built here against a number of other signed ranking models, including separate-layer PageRank models, signed random-walks through the graph, and signed link prediction methods.

## Tech Stack

| Tool | Role in project |
|---|---|
| Python | Main programming language. |
| pandas | Data parsing, cleaning, aggregation, feature table construction, and result summarization. |
| NumPy | Numerical support. |
| SciPy | Sparse matrix construction for the lifted signed PageRank computation. |
| NetworkX | Directed graph construction and ordinary PageRank computation. |
| scikit-learn | Train/test split, baseline model, logistic regression, random forest, evaluation metrics, ROC curve, precision-recall curve, and confusion matrix. |
| Matplotlib | Data visualization. |
| Docker | Reproducible computing environment. |
| Git/GitHub | Version control and final project delivery. |

The tech stack supported the project by making the workflow reproducible and modular. Python handled the full data science pipeline; NetworkX provided graph-specific methods; SciPy enabled sparse lifted-state signed PageRank computation; scikit-learn supported machine learning and evaluation; Matplotlib produced report-ready figures; Docker ensured the full workflow could be rerun in a consistent environment; and GitHub provided the final submission format.

## Reproducibility Instructions

### Local Python Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the full pipeline:

```powershell
python src\run_pipeline.py
```

### Docker Run

Build the Docker image:

```powershell
docker build -t pagerank-rfa-audit .
```

Run the full pipeline inside Docker:

```powershell
docker run --rm -it -v "${PWD}:/app" -w /app pagerank-rfa-audit
```

The Docker run executes:

```text
src/02_clean_data.py
src/03_build_graph.py
src/04_compute_features.py
src/05_make_figures.py
src/06_model_edges.py
src/07_advanced_figures.py
src/08_signed_pagerank_extension.py
```

## Expected Outputs

After a successful run, the repository produces:

```text
data/processed/rfa_votes_clean.csv
data/processed/rfa_support_edges.csv
data/processed/edge_features.csv
outputs/graph_summary.json
outputs/pagerank_scores.csv
outputs/top_25_pagerank_nodes.csv
outputs/model_metrics.csv
outputs/signed_pagerank_scores.csv
outputs/signed_pagerank_top_nodes.csv
outputs/signed_score_correlations.csv
figures/*.png
```

## LLM Assistance Statement

I used ChatGPT as a coding, debugging, and writing-support tool to help design the project structure. It was also useful for troubleshooting Docker/GitHub issues, as well as generating code review sessions and quizzes to prepare for everything that would be useful in preparation for development of the respective parts of the project.

## References

Leskovec, Jure, and Andrej Krevl. **SNAP Datasets: Stanford Large Network Dataset Collection**. Stanford Network Analysis Project, June 2014. https://snap.stanford.edu/data/

Stanford Network Analysis Project. **Wikipedia Requests for Adminship (with text)**. https://snap.stanford.edu/data/wiki-RfA.html

Jung, J., Jin, W., Sael, L., and Kang, U. **Random Walk-Based Ranking in Signed Social Networks**. Knowledge and Information Systems, 2020. https://doi.org/10.1007/s10115-019-01364-z

Babul, Shazia'Ayn, Yu Tian, and Renaud Lambiotte. **Strong and Weak Random Walks on Signed Networks**. Communications Physics, 2025. https://www.nature.com/articles/s44260-025-00027-1

Gu, Kun, Yong Liu, Fangfang Li, and Weiping Zhu. **Signed PageRank on Online Rating Systems**. Journal of Systems Science and Complexity, 2022. https://doi.org/10.1007/s11424-021-0124-2
