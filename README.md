# neurovision-ai

`neurovision-ai` is a FastAPI service that runs a lightweight, CPU-friendly visual neuroscience pipeline:

- **YOLOv8n detection** (fixed to 320x320 input)
- **CLIP semantic embedding**
- **GradCAM visualization**
- **Layer activation logging**
- **Anatomical brain region simulation** (`V1`, `V2`, `V4`, `Inferotemporal_Cortex`, `Prefrontal_Cortex`)

The service is designed for constrained environments (2GB RAM, 1 CPU, no GPU) with sequential execution and explicit tensor cleanup.

## API

### `GET /health`
Health check endpoint.

### `POST /analyze`
Upload an image (`multipart/form-data`) using `file` field.

Example:

```bash
curl -X POST http://localhost:10000/analyze \
  -F "file=@sample.jpg"
```

Response shape:

```json
{
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.8731,
      "bbox_xyxy": [11.2, 9.9, 120.8, 300.0]
    }
  ],
  "brain_activity": [
    {
      "region": "V1",
      "mean_activation": 0.12,
      "max_activation": 0.91
    }
  ],
  "decision_path": [
    "Visual features extracted in V1",
    "Shape features processed in V2",
    "Object representation formed in IT Cortex",
    "Semantic meaning refined in Prefrontal Cortex"
  ],
  "gradcam": "base64_heatmap"
}
```

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 10000
```

## Docker

```bash
docker build -t neurovision-ai .
docker run --rm -p 10000:10000 neurovision-ai
```

## Render deployment

This repository includes:

- `Dockerfile`
- `render.yaml`

On Render, create a new **Blueprint** or Web Service from this repo. Render will build the Docker image and start:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

## Notes on resource usage

- YOLO model is fixed to **YOLOv8n**.
- Input is resized to **320x320**.
- Inference is **sequential** to limit peak memory.
- `torch.no_grad()` is used for all non-GradCAM stages.
- Hooks are registered only on selected mapped layers.
