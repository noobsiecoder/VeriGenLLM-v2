# Secret Variables

This folder houses all sensitive (or enviroment) variables for:

- Runs open-source LLMs for evals.
- [Google Cloud Platform](https://cloud.google.com/) is used as cloud service.

---

### Folder structure

```
.
├── secrets/
│   ├── models-api.env     # Contains all proprietary LLM's API
│   ├── gcp-storage.json   # GCP secret key to run via client SDK
└── .. 
```

---

### Files

1. `models-api.env`:
   - API to access each proprietary LLM.

   ```
   CLAUDE_API=<your-claude-api-key-here>
   GEMINI_API=<your-gemini-api-key-here>
   OPENAI_API=<your-openai-api-key-here>
   HUGGINGFACE_TOKEN=<your-huggingface-token-here>
   ```

1. `gcp-storage.json`:
   - API access data downloaded from [service account - IAM and Admin tab](https://console.cloud.google.com/iam-admin/serviceaccounts).
   - Access GCP's storage bucket via VM instance.
   ```
   # To download:
   IAM and Admin > Service Acounts > Create Service Account > Set Role > Download JSON file
   ```

---
