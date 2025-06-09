# CRD X-Ray

A Kubernetes operator that analyzes Custom Resource Definitions (CRDs) and their controllers using LLM-powered analysis.

## Features

- Monitors all CRDs in a Kubernetes cluster
- Tracks CRD events and resources
- Identifies controllers using LLM analysis
- Stores data in SQLite and Qdrant vector database
- Provides natural language querying capabilities

## Prerequisites

- Python 3.8+
- Kubernetes cluster with kubectl configured
- Qdrant database instance
- Anthropic API key

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with the following variables:
```
ANTHROPIC_API_KEY=your_anthropic_api_key
QDRANT_URL=your_qdrant_url
QDRANT_PORT=your_qdrant_port
```

3. Run the operator:
```bash
python main.py
```

## Usage

The operator will automatically:
1. Discover all CRDs in the cluster
2. Collect events and resources for each CRD
3. Identify controllers using LLM analysis
4. Store data in SQLite and Qdrant
5. Enable natural language querying of the collected data

## Database Schema

### SQLite Tables

1. `crd_table`:
   - crd (TEXT, PRIMARY KEY)
   - last_updated_timestamp (DATETIME)
   - controller_name (TEXT)
   - names (TEXT)

2. `controller_table`:
   - controller (TEXT, PRIMARY KEY)
   - last_updated_timestamp (DATETIME)

3. `instance_table`:
   - resource_name (TEXT, PRIMARY KEY)
   - crd (TEXT, FOREIGN KEY)

### Qdrant Collections

1. `crd_data`: Stores CRD events and manifests
2. `resource_data`: Stores resource logs and manifests
3. `controller_data`: Stores controller logs and manifests

## Querying

The operator provides a natural language interface for querying the collected data. Example queries:

- "Show me all CRDs managed by the cert-manager controller"
- "What resources are associated with the prometheus CRD?"
- "List all events for the istio CRD"

## License

MIT 