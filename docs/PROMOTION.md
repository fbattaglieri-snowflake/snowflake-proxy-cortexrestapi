# Model Qualification & Zero-Touch Promotion

> **Disclaimer:** This project is an independent community / field-engineering implementation built on top of Snowflake Cortex REST APIs and Snowpark Container Services (SPCS). It is **NOT an official Snowflake product**.

## Step 1: Ensure Cross-Region Inference Is Enabled

Before qualifying or promoting models on your Snowflake account, verify that Cross-Region Inference is enabled (recommended: `ANY_REGION` where permitted by your compliance requirements):

```sql
SHOW PARAMETERS LIKE 'CORTEX_ENABLED_CROSS_REGION' IN ACCOUNT;
ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';
```

## Step 2: Qualify the Candidate Model

Use the companion skill and test runner [**`cortex-model-compatibility-matrix`**](https://github.com/fbattaglieri-snowflake/cortex-model-compatibility-matrix) to verify that the model passes text, streaming, single/parallel tool roundtrips, error handling, and multi-round tool ID checks on its target protocol (`chat`, `responses`, or `messages`).

## Step 3: Zero-Touch Promotion into the Registry (`scripts/promote_model.py`)

With `admin_approvers` configured in your Registry trust file (`registry-trust.json`) and `PROXY_ADMIN_APPROVERS` set on the Proxy deployment, you can promote a newly qualified model **without restarting or altering any SPCS service**:

```bash
python scripts/promote_model.py \
  --model <model-id> \
  --protocol <chat|responses|messages> \
  --account <YOUR_ACCOUNT_LOCATOR> \
  --upstream https://<your-org>-<your-account>.snowflakecomputing.com/api/v2/cortex/v1 \
  --artifact <64-hex-artifact-digest> \
  --expected-revision <current-registry-revision> \
  --approver <YOUR_ADMIN_IDENTITY> \
  --output-dir ./out-package \
  --stage-inbox @YOUR_DB.YOUR_SCHEMA.YOUR_STAGE/inbox \
  --connection <your_snow_cli_connection>
```

What happens automatically:
1. `scripts/promote_model.py` generates a `ProfileV2` descriptor (`normalize_client_options=True`, token range `1..128000`, `response_stream="ordered-items"` on Responses, 365-day validity under `administrative-standard-v1`) and uploads the content-addressed `<sha256>.json` + `.evidence` files to the Snowflake Stage inbox.
2. The `registry_service` background loop (`Publisher.scan()`, running every 30 seconds) validates the package, commits the new generation in SQLite, and writes a durable checkpoint.
3. Within another 30 seconds, all `llm_proxy_standard` serving replicas refresh their `SnapshotCache` and expose the model on `GET /<protocol>/v1/models` and `POST /<protocol>/v1/...` to **n8n**, **Hermes**, **Langflow**, and **LangChain**.
