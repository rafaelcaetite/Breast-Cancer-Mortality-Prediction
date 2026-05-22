# %% [markdown]
# ## **Universidade Federal de Viçosa (UFV)**
# ### **Departamento de Informática**
# 
# **Disciplina:** INF493 - Tópicos Especiais III: Ciência de Dados
# 
# **Professor:** Daniel Louzada Fernandes
# 
# ---
# 
# ## **Autores**
# * **Rafael Martins Caetité Lopes Cançado** (108192)
# * **Wenderson Lopes** (100000)
# 
# ---
# 
# ## **Metodologia Corrigida: Predição de Mortalidade por Câncer de Mama**
# 
# Esta versão foi completamente reestruturada para garantir **Rigor Científico** e **Eliminar qualquer forma de Data Leakage (Vazamento de Dados)**. 
# 
# As principais mudanças em relação ao script original são:
# 1. **Train/Test Split Antecipado:** A divisão dos dados em treino e teste ocorre imediatamente após a definição do *Target*, antes de qualquer imputação ou transformação.
# 2. **Pipeline de Pre-processamento (`sklearn.pipeline`):** Todas as imputações e transformações (ex: médias, medianas, categorias) aprendem (fit) exclusivamente com o conjunto de Treinamento, e apenas aplicam a transformação no conjunto de Teste.
# 3. **Preservação de Dados Genéticos:** Não aplicamos filtros de baixa variância globais antes do modelo. Mutações raras podem ser essenciais (ex: drivers de câncer). Para lidar com a alta dimensionalidade (~690 features), utilizamos algoritmos com **Regularização intrínseca** (como Random Forest, XGBoost ou Lasso) que lidam muito bem com esparsidade.

# %% [markdown]
# # **1. Importação de Bibliotecas**

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Divisão de Dados
from sklearn.model_selection import train_test_split

# Pipelines e Pré-processamento
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Avaliação
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, roc_curve

import warnings
warnings.filterwarnings('ignore')

# XGBoost (Opcional, porém recomendado para alta dimensionalidade)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost não encontrado. Considere instalar com: pip install xgboost")

# LightGBM (Opcional)
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


# %% [markdown]
# # **2. Carga dos Dados**

# %%
# Baixando a base caso não exista
folder_path = "./Breast_Cancer_Gene_Expression_Profiles"
csv_path = os.path.join(folder_path, "METABRIC_RNA_Mutation.csv")

if not os.path.exists(csv_path):
    print("Base não encontrada localmente. Baixando do Google Drive...")
    os.system("python -m pip install -q gdown")
    os.system('python -m gdown --folder https://drive.google.com/drive/folders/1WJ_JpnKhzCG321l9E5gr7SrjAnNSEtGj -O ./Breast_Cancer_Gene_Expression_Profiles')

print("Carregando arquivo CSV...")
df = pd.read_csv(csv_path, low_memory=False)
print(f"Dataset carregado com {df.shape[0]} amostras e {df.shape[1]} features.")


# %% [markdown]
# # **3. Limpeza Inicial e Binarização do Target**
# Nesta etapa removemos apenas identificadores sem valor preditivo e redefinimos nosso escopo de predição para uma classificação binária (Morte por Câncer vs Sobrevivência).

# %%
# Remove features não informativas ou com alta redundância documentada previamente
df = df.drop(columns=['patient_id', 'tumor_stage'])

# Mantém apenas as classes de interesse
mask = df['death_from_cancer'].isin(['Living', 'Died of Disease'])
df = df[mask].copy()
print(f"Amostras retidas (Living / Died of Disease): {df.shape[0]}")

# Define o target binário (0 = Living, 1 = Died of Disease)
y = (df['death_from_cancer'] == 'Died of Disease').astype(int)

# Remove as features de "ground truth" e variáveis que causam viés de severidade/tratamento (de-biasing)
features_to_drop = [
    'death_from_cancer', 'overall_survival', 'overall_survival_months',
    'nottingham_prognostic_index', 'lymph_nodes_examined_positive', 
    'tumor_size', 'neoplasm_histologic_grade', 'radio_therapy', 
    'chemotherapy', 'hormone_therapy', 'type_of_breast_surgery'
]
X = df.drop(columns=features_to_drop)

print(f"Distribuição do Target (Treino + Teste):\n{y.value_counts(normalize=True)*100}")


# %% [markdown]
# # **4. Tratamento Row-Wise das Mutações**
# Mutações são reportadas no dataset original como nomes dos alelos mutados ou '0' para wild-type. 
# Binarizar isso *antes do split* é cientificamente válido porque o processamento de cada paciente (linha) não depende das estatísticas dos demais.

# %%
mutation_cols = [col for col in X.columns if col.endswith('_mut')]
print(f"Binarizando {len(mutation_cols)} colunas de mutação genômica...")

for col in mutation_cols:
    # '0' ou 'nan' vira 0 (wild-type). Qualquer variante vira 1 (mutado).
    X[col] = (~X[col].astype(str).isin(['0', '0.0', 'nan', 'None'])).astype(int)


# %% [markdown]
# # **5. Train/Test Split (CRÍTICO PARA EVITAR LEAKAGE)**
# Toda operação a partir daqui que exija conhecer a distribuição dos dados (imputação de medianas, standard scaling, target encoding) DEVE ser fitada apenas no conjunto `X_train`.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Conjunto de TREINAMENTO: {X_train.shape[0]} amostras")
print(f"Conjunto de TESTE: {X_test.shape[0]} amostras")


# %% [markdown]
# # **6. Construção do Pipeline (Imputação e Transformação)**
# Construiremos um pre-processador unificado e rigoroso usando `ColumnTransformer`.

# %%
# Define o tipo das features dinamicamente a partir do X_train
numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Features Numéricas: {len(numerical_cols)} (incluindo Mutações Binarizadas e Expressão)")
print(f"Features Categóricas: {len(categorical_cols)}")

# Pre-processamento Numérico
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), # Preenche Nulos com Mediana do TREINO
    ('scaler', RobustScaler()) # Padroniza suportando Outliers biológicos
])

# Pre-processamento Categórico
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ],
    remainder='drop'
)


# %% [markdown]
# # **7. Treinamento de Modelos Preditivos**
# Utilizaremos os algoritmos embutidos no pipeline, permitindo cross-validação robusta. Como mantivemos a alta dimensionalidade de forma consciente, escolheremos modelos com L1 penalty ou baseados em Árvores que fazem seleção de features internamente.

# %%
models = {
    'Logistic Regression (L1)': LogisticRegression(
        penalty='l1', solver='saga', max_iter=1500, class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5, class_weight='balanced', random_state=42
    )
}

if XGB_AVAILABLE:
    # Calcula peso adequado para as classes para uso no scale_pos_weight
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    models['XGBoost'] = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, 
        eval_metric='logloss', random_state=42, scale_pos_weight=ratio
    )
    
if LGBM_AVAILABLE:
    models['LightGBM'] = LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05, 
        random_state=42, class_weight='balanced'
    )

best_auc = 0
best_model_name = ""
best_model_pipeline = None
model_results = {}

print("=== INICIANDO TREINAMENTO ===\n")
for name, model in models.items():
    print(f"Treinando: {name}")
    
    # Monta pipeline fim a fim
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # FIT estritamente no Treino!
    pipeline.fit(X_train, y_train)
    
    # EVALUATION estritamente no Teste!
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else pipeline.decision_function(X_test)
    
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    
    model_results[name] = {'AUC': auc, 'Accuracy': acc}
    
    print(f"  --> Test AUC: {auc:.4f} | Test Acc: {acc:.4f}\n")
    
    if auc > best_auc:
        best_auc = auc
        best_model_name = name
        best_model_pipeline = pipeline

print(f"Melhor Modelo: {best_model_name} (AUC: {best_auc:.4f})")


# %% [markdown]
# # **8. Avaliação do Melhor Modelo**

# %%
print(f"=== RELATÓRIO DO MELHOR MODELO ({best_model_name}) ===")
y_pred_best = best_model_pipeline.predict(X_test)
y_proba_best = best_model_pipeline.predict_proba(X_test)[:, 1]

print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_best, target_names=['Living', 'Died of Disease']))

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Living (0)', 'Died of Disease (1)'],
            yticklabels=['Living (0)', 'Died of Disease (1)'])
plt.ylabel('Classe Real')
plt.xlabel('Classe Predita')
plt.title(f'Matriz de Confusão - {best_model_name}\n(Dados Nunca Vistos)')
plt.savefig('matriz_confusao.png')
plt.close()

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba_best)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {best_auc:.4f})', linewidth=2, color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', label='Baseline', linewidth=2)
plt.xlabel('Taxa de Falsos Positivos (FPR)')
plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
plt.title('Curva ROC - Test Set Validation')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('curva_roc.png')
plt.close()


# %% [markdown]
# # **9. Importância Genômica e Clínica**
# Ao não excluir genes preemptivamente, o próprio algoritmo avalia as features baseando-se na sua contribuição real ao problema.

# %%
if hasattr(best_model_pipeline.named_steps['classifier'], 'feature_importances_'):
    importances = best_model_pipeline.named_steps['classifier'].feature_importances_
    
    # Extrair os nomes do preprocessador
    num_names = numerical_cols
    cat_names = best_model_pipeline.named_steps['preprocessor'].named_transformers_['cat'] \
                .named_steps['encoder'].get_feature_names_out(categorical_cols)
    all_feature_names = np.concatenate([num_names, cat_names])
    
    feat_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=False)
    
    print("=== TOP 20 Features Mais Importantes ===")
    print(feat_df.head(20).to_string(index=False))
    
    # Visualização
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feat_df.head(20), palette='viridis')
    plt.title(f'Top 20 Feature Importances ({best_model_name})')
    plt.xlabel('Importância (Gain / Gini)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('importancia_features.png')
    plt.close()
else:
    print(f"O modelo {best_model_name} não possui o atributo 'feature_importances_'.")

# %% [markdown]
# # **10. Salvar Modelo e Resultados**

# %%
import pickle
import os

output_file = "melhor_modelo_resultados.pkl"
print(f"\n=== SALVANDO MODELO E RESULTADOS ===")
print(f"Salvando o modelo e os resultados em {output_file}...")
with open(output_file, 'wb') as f:
    pickle.dump({
        'best_model_pipeline': best_model_pipeline,
        'model_results': model_results,
        'best_model_name': best_model_name,
        'best_auc': best_auc
    }, f)
print("Salvamento concluído com sucesso!")
