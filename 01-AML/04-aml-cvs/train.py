# ---------------------------------------------------------------------
# train.py  (compatible con MLTable en modo DIRECT)
# ---------------------------------------------------------------------
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import subprocess
import sys

import subprocess
import sys

try:
    import mltable
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mltable"])
    import mltable  # IMPORTANTE: volver a importar después de la instalación

def load_dataframe(uri_or_path: str) -> pd.DataFrame:
    """
    Carga un MLTable:
      • Si es una URI azureml:// (modo DIRECT) → usa mltable.load().
      • Si es un directorio local montado/descargado → busca el primer .csv.
    """
    if uri_or_path.startswith("azureml://"):
        print("🗄️  Cargando MLTable en modo DIRECT:", uri_or_path)
        tbl = mltable.load(uri_or_path)          # Devuelve objeto MLTable
        df = tbl.to_pandas_dataframe()           # <- DataFrame final
        return df.reset_index(drop=True)

    # Modo MOUNT / DOWNLOAD
    print("📂  Cargando dataset desde directorio local:", uri_or_path)
    csv_files = [f for f in os.listdir(uri_or_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(
            f"No se encontró ningún .csv dentro de {uri_or_path}"
        )
    csv_path = os.path.join(uri_or_path, csv_files[0])
    print("   → Usando archivo:", csv_path)
    return pd.read_csv(csv_path)

def main() -> None:
    # ---------------------------------------------------------------
    # 1. Argumentos
    # ---------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_asset",
        type=str,
        required=True,
        help="URI azureml://… o ruta local del MLTable"
    )
    args, _ = parser.parse_known_args()

    # ---------------------------------------------------------------
    # 2. Carga de datos
    # ---------------------------------------------------------------
    df = load_dataframe(args.data_asset)

    # ---------------------------------------------------------------
    # 3. Ingeniería de características
    # ---------------------------------------------------------------
    df["target"] = df["decision_historica"].map(
        {"Aceptado": 1, "Rechazado": 0}
    )

    text_features = "texto_cv"
    categorical_features = ["puesto_solicitado", "universidad_origen"]
    numeric_features = ["años_experiencia"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(), text_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop"
    )

    X = df[numeric_features + categorical_features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    # ---------------------------------------------------------------
    # 4. Modelo
    # ---------------------------------------------------------------
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                random_state=42, solver="liblinear"))
        ]
    )

    # ---------------------------------------------------------------
    # 5. Entrenamiento y métricas
    # ---------------------------------------------------------------
    mlflow.start_run()

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"✅  Accuracy: {acc:.4f}")
    print(f"✅  F1‑score: {f1:.4f}")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # ---------------------------------------------------------------
    # 6. Registro del modelo
    # ---------------------------------------------------------------
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name="modelo-cv-adaptado",
        input_example=X_train.iloc[:1],
        signature=mlflow.models.infer_signature(X_train, y_train),
    )

    mlflow.end_run()
    print("🏁  Entrenamiento terminado.")

if __name__ == "__main__":
    main()
