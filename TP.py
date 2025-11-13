# ---------------------------------------------------
# PARTE 1 - LEITURA, LIMPEZA E NORMALIZAÇÃO
# ---------------------------------------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1) LER O DATASET
df = pd.read_csv("spotify_songs.csv")

print(df.shape)
print(df.head())
print(df.columns)
print(df.info())

# 2) TIRAR COLUNAS INÚTEIS
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 3) ESCOLHER FEATURES ÚTEIS
cols_to_keep = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_ms', 'track_popularity'
]

# manter só as que existem mesmo neste dataset
cols_to_keep = [c for c in cols_to_keep if c in df.columns]

df_features = df[cols_to_keep].copy()

print("\nColunas que vamos usar:")
print(df_features.columns)

# 4) LIMPAR NaN E DUPLICADOS
print("\nValores em falta por coluna:")
print(df_features.isna().sum())

df_features = df_features.dropna().drop_duplicates()
print("\nDepois de limpar NaN e duplicados:", df_features.shape)

# 5) FILTRAR OUTLIERS SIMPLES
if 'tempo' in df_features.columns:
    df_features = df_features[(df_features['tempo'] > 40) & (df_features['tempo'] < 220)]
if 'duration_ms' in df_features.columns:
    df_features = df_features[(df_features['duration_ms'] > 30000) & (df_features['duration_ms'] < 600000)]

print("\nDepois de filtrar tempos/durações estranhos:", df_features.shape)

# manter uma cópia do original correspondente a estas linhas
df_clean = df.loc[df_features.index].copy()

# 6) NORMALIZAR
scaler = StandardScaler()
df_scaled = df_features.copy()
df_scaled[df_features.columns] = scaler.fit_transform(df_features[df_features.columns])

print("\nPrimeiras linhas já normalizadas:")
print(df_scaled.head())

# 7) GUARDAR LIMPO
df_scaled.to_csv("spotify_clean.csv", index=False)
print("\nFicheiro 'spotify_clean.csv' guardado.")

# 8) K-MEANS
kmeans = KMeans(n_clusters=5, random_state=42)
df_scaled['cluster'] = kmeans.fit_predict(df_scaled[df_features.columns])

print("\nClusters atribuídos:")
print(df_scaled['cluster'].value_counts())

# 9) JUNTAR CLUSTER À INFO DAS MÚSICAS
df_result = df_clean.join(df_scaled['cluster'])
print(df_result[['track_name', 'track_artist', 'playlist_genre', 'cluster']].head(20))

# ---------------------------------------------------
# PARTE 2 - CLASSIFICAÇÃO (Random Forest e Decision Tree)
# ---------------------------------------------------


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  # já tens em cima, mas fica aqui se correr separado

print("\n========================")
print("INÍCIO DA CLASSIFICAÇÃO")
print("========================")

# 1️⃣ CRIAR O RÓTULO (AGORA COM LIMIAR 70 EM VEZ DE 60)
#   1 → música muito popular
#   0 → restante
df_scaled['popular_label'] = (df_features['track_popularity'] > 70).astype(int)

# 2️⃣ VER SE A CLASSE ESTÁ DESBALANCEADA
print("\nDistribuição das classes (0 = não popular, 1 = popular):")
print(df_scaled['popular_label'].value_counts())

# 3️⃣ DEFINIR FEATURES (X) E TARGET (y)
X = df_scaled.drop(['track_popularity', 'cluster', 'popular_label'], axis=1, errors='ignore')
y = df_scaled['popular_label']

# 4️⃣ TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# MODELO 1 — DECISION TREE (um bocadinho mais controlado)
# =========================================================
dt_model = DecisionTreeClassifier(
    random_state=42,
    class_weight='balanced',   # novo
    max_depth=10               # para não ficar tão “doido”
)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\n--- Decision Tree ---")
print("Acurácia:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# =========================================================
# MODELO 2 — RANDOM FOREST (melhorado)
# =========================================================
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced',   # novo: dá mais peso à classe 1
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest ---")
print("Acurácia:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# =========================================================
# MATRIZES DE CONFUSÃO
# =========================================================
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title('Decision Tree')
ax[0].set_xlabel('Previsto')
ax[0].set_ylabel('Real')

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens', ax=ax[1])
ax[1].set_title('Random Forest')
ax[1].set_xlabel('Previsto')
ax[1].set_ylabel('Real')

plt.tight_layout()
plt.show()

# =========================================================
# IMPORTÂNCIA DAS FEATURES
# =========================================================
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nImportância das variáveis no modelo Random Forest:")
print(importances)

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=importances.index, palette="viridis")
plt.title("Importância das Features (Random Forest)")
plt.xlabel("Importância")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# =========================================================
# PARTE 3 – REGRAS DE ASSOCIAÇÃO (APRIORI)
# =========================================================

from mlxtend.frequent_patterns import apriori, association_rules

print("\n========================")
print("REGRAS DE ASSOCIAÇÃO (APRIORI)")
print("========================")

# ponto de partida: usar só as colunas de áudio + popularidade
df_rules = df_features.copy()

# 1) discretizar (transformar contínuos em 0/1)
cols_to_bin = ['danceability', 'energy', 'valence', 'acousticness', 'loudness']

for col in cols_to_bin:
    med = df_rules[col].median()
    df_rules[col] = (df_rules[col] > med).astype(int)

# 2) criar coluna de alvo binário (popular ou não)
df_rules['popular'] = (df_features['track_popularity'] > 70).astype(int)

# 3) agora selecionar só as colunas binárias (as que acabámos de criar)
df_rules_bin = df_rules[cols_to_bin + ['popular']].copy()

# garantir que são todas int/bool
df_rules_bin = df_rules_bin.astype(int)

# 4) aplicar apriori
frequent_itemsets = apriori(df_rules_bin, min_support=0.1, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# ordenar por confiança para veres as melhores
rules = rules.sort_values(by='confidence', ascending=False)

print("\nTop 10 regras de associação encontradas:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# =========================================================
# PARTE 4 – FUSÃO COM MOODIFY (EMOÇÕES) POR track_id
# =========================================================
import pandas as pd
import numpy as np
import re

print("\n========================")
print("FUSÃO COM MOODIFY (EMOÇÕES)")
print("========================")

# 1) LER O FICHEIRO COM URI (tem as emoções em 'labels')
mood_uri_path = "278k_labelled_uri.csv"  # ajusta se o nome for diferente
df_mood = pd.read_csv(mood_uri_path)

# 2) LIMPAR COLUNAS INÚTEIS E NORMALIZAR NOMES
drop_cols = [c for c in df_mood.columns if c.lower().startswith("unnamed")]
df_mood = df_mood.drop(columns=drop_cols, errors='ignore')
df_mood.columns = df_mood.columns.str.strip().str.lower()

# 3) VALIDAR COLUNAS NECESSÁRIAS
if "uri" not in df_mood.columns or "labels" not in df_mood.columns:
    raise ValueError("O ficheiro Moodify precisa de ter as colunas 'uri' e 'labels'.")

# 4) EXTRair track_id do uri (spotify:track:<id> -> <id>)
def extract_track_id(uri):
    try:
        return str(uri).split(":")[-1]
    except Exception:
        return np.nan

df_mood["track_id"] = df_mood["uri"].apply(extract_track_id)
df_mood = df_mood.dropna(subset=["track_id"]).copy()

# 5) MANTER SÓ O QUE INTERESSA PARA O MERGE
df_mood_small = df_mood[["track_id", "labels"]].drop_duplicates()

# 6) (Opcional) mapear labels numéricos para nomes – CONFIRMA NO README DO DATASET!
# >>> ATENÇÃO: verifica no Kaggle a semântica exata. Exemplo-tipo:
# 0=calm, 1=happy, 2=sad, 3=energetic   (AJUSTA SE FOR DIFERENTE!)
label_map = {0: "calm", 1: "happy", 2: "sad", 3: "energetic"}
df_mood_small["emotion"] = df_mood_small["labels"].map(label_map)

# 7) MERGE COM O TEU RESULTADO (df_result tem 'track_id' vindo do Spotify)
if "track_id" not in df_result.columns:
    # Se não tiveres track_id no df_result, tenta criá-lo a partir de possíveis colunas:
    # (no teu dataset original já existe 'track_id', por isso isto é apenas salvaguarda)
    raise ValueError("df_result não tem 'track_id'. Garante que preservaste a coluna 'track_id' do CSV original.")

df_result = df_result.merge(df_mood_small[["track_id", "labels", "emotion"]],
                            on="track_id", how="left")

# 8) RELATÓRIO DE COBERTURA
tot = len(df_result)
com_emocao = df_result["emotion"].notna().sum()
print(f"\nCobertura da fusão (músicas com emoção): {com_emocao}/{tot} = {com_emocao/tot:.1%}")

# 9) DISTRIBUIÇÕES ÚTEIS
print("\nTop 10 emoções (contagem):")
print(df_result["emotion"].value_counts().head(10))

print("\nEmoção por cluster (amostra):")
print(df_result.groupby("cluster")["emotion"].value_counts().head(20))

# 10) GERAR PLAYLISTS PARA TODAS AS EMOÇÕES EXISTENTES
print("\n🎧 A gerar playlists por emoção...")

for alvo in df_result["emotion"].dropna().unique():
    cand = df_result[df_result["emotion"].fillna("").str.lower() == alvo.lower()]
    if len(cand) >= 10:
        playlist_emocao = cand.sample(10, random_state=42)
        print(f"\n🎵 Playlist para emoção = '{alvo}' ({len(cand)} músicas disponíveis):")
        print(playlist_emocao[["track_name", "track_artist", "playlist_genre", "emotion"]])
        playlist_emocao[["track_name", "track_artist", "playlist_genre", "emotion"]].to_csv(
            f"playlist_{alvo.lower()}.csv", index=False
        )
        print(f"Ficheiro 'playlist_{alvo.lower()}.csv' guardado.")
    else:
        print(f"\n⚠️ Não há músicas suficientes com emoção '{alvo}' ({len(cand)} apenas).")

# =========================================================
# IMPUTAÇÃO DE EMOÇÃO PARA MÚSICAS SEM RÓTULO (KNN)
# =========================================================
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("\n========================")
print("IMPUTAÇÃO DE EMOÇÃO COM KNN (Moodify features)")
print("========================")

emo_feats = ['danceability','energy','loudness','speechiness','acousticness',
             'instrumentalness','liveness','valence','tempo','duration_ms']

df_mood_names = pd.read_csv("278k_song_labelled.csv")
df_mood_names.columns = df_mood_names.columns.str.lower().str.strip()

has_cols = all(c in df_mood_names.columns for c in emo_feats+['labels'])
if not has_cols:
    print("⚠️  KNN ignorado: 278k_song_labelled.csv não tem todas as colunas necessárias.")
else:
    X_train_emo = df_mood_names[emo_feats].copy()
    y_train_emo = df_mood_names['labels'].copy()

    label_map_knn = {0:'calm', 1:'happy', 2:'sad', 3:'energetic'}
    if y_train_emo.dtype.kind in "iu":
        y_train_emo = y_train_emo.map(label_map_knn).fillna('unknown').astype(str)
    else:
        y_train_emo = y_train_emo.astype(str).str.lower().str.strip()

    miss_mask = df_result['emotion'].isna()
    to_fill = df_result.loc[miss_mask, :].copy()

    feat_cols_in_result = [c for c in emo_feats if c in to_fill.columns]

    if len(feat_cols_in_result) < 5:
        print("⚠️  KNN ignorado: faltam features suficientes no df_result para prever emoção.")
    else:
        knn = Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('model', KNeighborsClassifier(n_neighbors=11, weights='distance'))
        ])
        knn.fit(X_train_emo[feat_cols_in_result], y_train_emo)

        preds = knn.predict(to_fill[feat_cols_in_result])
        df_result.loc[miss_mask, 'emotion'] = preds

        tot = len(df_result); com = df_result['emotion'].notna().sum()
        print(f"✔️  Cobertura após imputação KNN: {com}/{tot} = {com/tot:.1%}")


#=========================================================
# AVALIAR K (ELBOW + SILHOUETTE)
# =========================================================
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt

Ks = range(2, 9)
inertias, silhouettes = [], []

for k in Ks:
    km = KMeans(n_clusters=k, random_state=42)
    labs = km.fit_predict(df_scaled[df_features.columns])
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(df_scaled[df_features.columns], labs))

plt.figure(figsize=(8,4))
plt.plot(Ks, inertias, marker='o'); plt.title("Elbow (Inertia)"); plt.xlabel("k"); plt.ylabel("inertia")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
plt.plot(Ks, silhouettes, marker='o'); plt.title("Silhouette"); plt.xlabel("k"); plt.ylabel("score")
plt.tight_layout(); plt.show()

# =========================================================
# GRID SEARCH – RANDOM FOREST
# =========================================================
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "class_weight": ["balanced"]  # manter o balancing
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=1
)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
print("\nMelhor RF:", rf_grid.best_params_)

y_pred_best = best_rf.predict(X_test)
print("\n--- Random Forest (TUNADO) ---")
print("Acurácia:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

# =========================================================
# PCA 2D dos clusters
# =========================================================
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
Z = pca.fit_transform(df_scaled[df_features.columns])
plt.figure(figsize=(8,6))
plt.scatter(Z[:,0], Z[:,1], c=df_scaled['cluster'], s=6)
plt.title("Clusters de músicas (PCA 2D)"); plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout(); plt.show()

# =========================================================
# Barras: nº de músicas por emoção
# =========================================================
emo_counts = df_result['emotion'].value_counts().sort_values(ascending=False)
plt.figure(figsize=(7,4))
emo_counts.plot(kind='bar')
plt.title("Distribuição de emoções (Moodify)"); plt.xlabel("Emoção"); plt.ylabel("Contagem")
plt.tight_layout(); plt.show()

# =========================================================
# 2º PASSO DE FUSÃO POR (track_name + track_artist) NORMALIZADOS — ROBUSTO
# =========================================================
import re

def norm(s):
    s = str(s).lower().strip()
    s = s.replace("&","and")
    s = re.sub(r'[^a-z0-9]+', '', s)
    return s

def pick_col(cols, exact_first, contains_patterns):
    """Tenta primeiro correspondência exata (por lower()),
    depois procura colunas que contenham todas as palavras de um padrão."""
    cols_low = [c.lower() for c in cols]
    # exatas
    for name in exact_first:
        if name in cols_low:
            return cols[cols_low.index(name)]
    # por padrões de substrings
    for pats in contains_patterns:
        for c in cols:
            cl = c.lower()
            if all(p in cl for p in pats):
                return c
    return None

try:
    # Só tentar fallback nas linhas sem emoção depois do merge por ID
    left_missing = df_result[df_result['emotion'].isna()].copy()
    if left_missing.empty:
        print("Sem linhas em falta de emoção — fallback name+artist desnecessário.")
    else:
        left_missing['__key__'] = (left_missing['track_name'].apply(norm) + "_" +
                                   left_missing['track_artist'].apply(norm))

        df_mood_names = pd.read_csv("278k_song_labelled.csv")
        df_mood_names.columns = df_mood_names.columns.str.lower().str.strip()

        # Descobrir colunas candidatas de forma segura
        nm = pick_col(
            df_mood_names.columns,
            exact_first=['track_name', 'song_name', 'name', 'title'],
            contains_patterns=[['track', 'name'], ['song', 'name']]
        )
        na = pick_col(
            df_mood_names.columns,
            exact_first=['track_artist', 'artist', 'artist_name', 'singer'],
            contains_patterns=[['track', 'artist'], ['artist']]
        )
        lab = pick_col(
            df_mood_names.columns,
            exact_first=['labels','label','emotion','mood','class'],
            contains_patterns=[['label'], ['emotion'], ['mood']]
        )

        if not nm or not na or not lab:
            print("⚠️  Fallback por nome+artista ignorado: não encontrei colunas compatíveis no 278k_song_labelled.csv")
            print("Colunas disponíveis:", list(df_mood_names.columns))
        else:
            df_mood_names['__key__'] = (df_mood_names[nm].apply(norm) + "_" +
                                        df_mood_names[na].apply(norm))
            df_mood_names_small = df_mood_names[['__key__', lab]].drop_duplicates()

            # map labels -> texto (ajusta se a legenda for diferente no teu dataset)
            label_map2 = {0: "calm", 1: "happy", 2: "sad", 3: "energetic"}
            # Se já vier texto, mantém; se for número, mapeia:
            if df_mood_names_small[lab].dtype.kind in "iu":
                df_mood_names_small['emotion_fallback'] = df_mood_names_small[lab].map(label_map2)
            else:
                df_mood_names_small['emotion_fallback'] = df_mood_names_small[lab].astype(str).str.lower().str.strip()

            merged_fb = left_missing.merge(df_mood_names_small[['__key__','emotion_fallback']],
                                           on='__key__', how='left')
            mask = df_result['emotion'].isna()
            df_result.loc[mask, 'emotion'] = merged_fb['emotion_fallback'].values

            tot = len(df_result); com = df_result['emotion'].notna().sum()
            print(f"Cobertura após fallback name+artist: {com}/{tot} = {com/tot:.1%}")

except Exception as e:
    print("Fallback name+artist ignorado por erro controlado:", e)

# =========================================================
# GUARDAR MODELOS E SCALER
# =========================================================
import joblib
joblib.dump(scaler, "scaler.joblib")
joblib.dump(kmeans, "kmeans.joblib")
joblib.dump(best_rf if 'best_rf' in globals() else rf_model, "rf_model.joblib")
print("Modelos e scaler guardados em disco.")

# =========================================================
# CLI: PERGUNTAR AO UTILIZADOR E GERAR PLAYLIST
# =========================================================
try:
    print("\n--- Gerar playlist por perfil ---")
    d = float(input("Danceability (0-1): ") or 0.6)
    e = float(input("Energy (0-1): ") or 0.7)
    v = float(input("Valence (0-1): ") or 0.6)
    emo = input("Emoção (happy/sad/calm/energetic ou Enter p/ ignorar): ").strip().lower()

    # construir vetor e escalar
    user_profile = { 'danceability': d, 'energy': e, 'valence': v }
    user_vec = pd.DataFrame([np.zeros(len(df_features.columns))], columns=df_features.columns)
    for k,vv in user_profile.items():
        if k in user_vec.columns: user_vec[k] = vv
    user_vec_sc = pd.DataFrame(scaler.transform(user_vec), columns=user_vec.columns)

    user_cluster = kmeans.predict(user_vec_sc[df_features.columns])[0]
    base = df_result[df_result['cluster']==user_cluster]
    if emo:
        base = base[base['emotion'].fillna("").str.lower()==emo]
    if len(base) >= 10:
        pl = base.sample(10, random_state=42)
        print("\n🎧 Playlist gerada:")
        print(pl[['track_name','track_artist','playlist_genre','emotion']])
        pl[['track_name','track_artist','playlist_genre','emotion']].to_csv("playlist_custom.csv", index=False)
        print("Guardado em 'playlist_custom.csv'.")
    else:
        print("Não há músicas suficientes com esses critérios.")
except Exception as ex:
    print("Input ignorado (execução non-interactive).", ex)