# Secret Variables

This folder houses all sensitive (or enviroment) variables for:

- Runs open-source LLMs for evals.
- [Google Cloud Platform](https://cloud.google.com/) is used for cloud service.

---

### Folder structure

```
.
├── secrets/
│   ├── gcp-storage.json   # GCP secret key to run via client SDK
│   ├── models-api.env     # Contains all proprietary LLM's API
│   └── ...
└── ... 
```

---

### Files

1. `gcp-storage.json`:
   - API access data downloaded from [service account - IAM and Admin tab](https://console.cloud.google.com/iam-admin/serviceaccounts).
   - Access GCP's storage bucket via VM instance.
    ```
    # To download:
    IAM and Admin > Service Acounts > Create Service Account > Set Role > Download JSON file
    ```

1. `models-api.env`:
   - API to access each proprietary LLM.

   ```env
   ANTHROPIC_API_KEY=<your-claude-api-key-here>
   GEMINI_API_KEY=<your-gemini-api-key-here>
   HUGGINGFACE_API_KEY=<your-huggingface-token-here>
   OPENAI_API_KEY=<your-openai-api-key-here>
   WANDB_API_KEY=<your-weight&biases-api-key-here>
   ```

---