# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Sarvam-30B on Databricks Model Serving
# MAGIC
# MAGIC This notebook registers the Sarvam-30B open-source model (Apache 2.0) from HuggingFace
# MAGIC and deploys it on Databricks Model Serving with GPU provisioned throughput.
# MAGIC
# MAGIC **Sarvam-30B**: 30B parameter MoE model (2.4B active), trained on 16T tokens,
# MAGIC supports 22 Indian languages + English. Ideal for sovereign AI deployments in India.

# COMMAND ----------

# MAGIC %pip install mlflow>=2.12.0 transformers torch accelerate
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import requests
import json
import time

# Configuration
CATALOG = "sarbanimaiti_catalog"
SCHEMA = "sarvam_voice_agent"
MODEL_NAME = "sarvam_30b"
ENDPOINT_NAME = "sarvam-30b-serving"
HF_MODEL_ID = "sarvamai/sarvam-30b"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Create catalog and schema

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Register model from HuggingFace to Unity Catalog

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
uc_model_name = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# Register the HuggingFace model using MLflow transformers flavor
with mlflow.start_run():
    model_info = mlflow.transformers.log_model(
        transformers_model=HF_MODEL_ID,
        artifact_path="model",
        task="llm/v1/chat",
        registered_model_name=uc_model_name,
        metadata={"source": "huggingface", "model_id": HF_MODEL_ID}
    )

print(f"Model registered: {uc_model_name}")
print(f"Model version: {model_info.registered_model_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Deploy to Model Serving endpoint with GPU
# MAGIC
# MAGIC Sarvam-30B is an MoE model with 30B total / 2.4B active parameters.
# MAGIC It can be served efficiently on a single A100 GPU.

# COMMAND ----------

host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Check if endpoint already exists
existing = requests.get(f"{host}/api/2.0/serving-endpoints/{ENDPOINT_NAME}", headers=headers)

if existing.status_code == 200:
    print(f"Endpoint '{ENDPOINT_NAME}' already exists. Updating...")
    # Update the endpoint
    update_payload = {
        "served_entities": [
            {
                "entity_name": uc_model_name,
                "entity_version": model_info.registered_model_version,
                "min_provisioned_throughput": 0,
                "max_provisioned_throughput": 1000,
                "scale_to_zero_enabled": True
            }
        ]
    }
    response = requests.put(
        f"{host}/api/2.0/serving-endpoints/{ENDPOINT_NAME}/config",
        headers=headers,
        json=update_payload
    )
else:
    print(f"Creating new endpoint '{ENDPOINT_NAME}'...")
    create_payload = {
        "name": ENDPOINT_NAME,
        "config": {
            "served_entities": [
                {
                    "entity_name": uc_model_name,
                    "entity_version": model_info.registered_model_version,
                    "min_provisioned_throughput": 0,
                    "max_provisioned_throughput": 1000,
                    "scale_to_zero_enabled": True
                }
            ]
        }
    }
    response = requests.post(
        f"{host}/api/2.0/serving-endpoints",
        headers=headers,
        json=create_payload
    )

print(f"Response: {response.status_code}")
print(json.dumps(response.json(), indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Wait for endpoint to be ready

# COMMAND ----------

print(f"Waiting for endpoint '{ENDPOINT_NAME}' to be ready...")
while True:
    resp = requests.get(f"{host}/api/2.0/serving-endpoints/{ENDPOINT_NAME}", headers=headers)
    state = resp.json().get("state", {}).get("ready", "NOT_READY")
    config_update = resp.json().get("state", {}).get("config_update", "NOT_UPDATING")
    print(f"  State: ready={state}, config_update={config_update}")
    if state == "READY" and config_update != "IN_PROGRESS":
        print("Endpoint is READY!")
        break
    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Test the endpoint

# COMMAND ----------

test_payload = {
    "messages": [
        {"role": "system", "content": "You are a helpful travel assistant for Indian travelers. Respond in the same language as the user. Be concise and helpful."},
        {"role": "user", "content": "मुझे गोवा जाना है, कुछ tips दो"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
}

response = requests.post(
    f"{host}/serving-endpoints/{ENDPOINT_NAME}/invocations",
    headers=headers,
    json=test_payload
)

print("Test response:")
print(json.dumps(response.json(), indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Endpoint Details
# MAGIC
# MAGIC Your Sarvam-30B model is now deployed on Databricks Model Serving.
# MAGIC
# MAGIC - **Endpoint URL**: `{host}/serving-endpoints/sarvam-30b-serving/invocations`
# MAGIC - **Model**: Sarvam-30B (Apache 2.0, 22 Indian languages + English)
# MAGIC - **Architecture**: MoE, 30B total / 2.4B active per token
# MAGIC - **Use in the voice agent app**: The Databricks App will call this endpoint for LLM inference
