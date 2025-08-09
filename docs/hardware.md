# Hardware in Cloud

## Google Cloud Platform Setup

### Instance Configuration

|      Type      | Specification  |
| :------------: | :------------: |
| Cloud Platform |  Google Cloud  |
|    Instance    | g2-standard-16 |
|      GPU       |  NVIDIA V100   |
|   GPU Memory   |     16 GB      |
|      Zone      | asia-east1-\*  |

### Resource Allocation

|    Resource    | Size |
| :------------: | ---: |
|      vCPU      |   16 |
|    RAM (GB)    |   64 |
| Disk Size (GB) |  350 |
|   Disk Type    |  SSD |

### GPU Details

- **Model**: NVIDIA Tesla V100
- **VRAM**: 16 GB HBM2
- **Compute Capability**: 7.0
- **CUDA Cores**: 5,120
- **Use Case**: Ideal for 7B parameter models

### Cost Optimization

- **Spot Instances**: ~70% cost reduction
- **Preemptible**: Suitable for batch inference
- **Region**: us-central1 (lowest GPU pricing)

### Setup Commands

```bash
# Create instance
gcloud compute instances create verigen-llm \
  --zone=asia-east1-* \
  --machine-type=g2-standard-16 \
  --accelerator=count=1,type=nvidia-tesla-v100 \
  --boot-disk-size=350GB \
  --boot-disk-type=pd-ssd

# SSH into instance
gcloud compute ssh verigen-llm --zone=us-central1-a
```

### Performance Notes

- **Model Loading**: ~2-3 minutes for 7B models
- **Inference Speed**: ~50-100 tokens/second
- **Batch Processing**: Optimal batch size = 8-16
- **Memory Usage**: ~13GB per 7B model (FP16)
