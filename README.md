# 🎵 Audio Cleaner Web

[![Build Status](https://github.com/BenjaminDanker/Audio-Cleaner-Web/actions/workflows/ci.yml/badge.svg)](https://github.com/BenjaminDanker/Audio-Cleaner-Web/actions)  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  [![Version](https://img.shields.io/github/v/release/BenjaminDanker/Audio-Cleaner-Web)](https://github.com/BenjaminDanker/Audio-Cleaner-Web/releases)

AI-powered video noise reduction in the cloud. Upload a video and receive a version with crystal-clear audio using state-of-the-art DeepFilterNet3.

## Table of Contents

- [🎵 Audio Cleaner Web](#-audio-cleaner-web)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Project Structure](#project-structure)
  - [Architecture](#architecture)
  - [Installation \& Local Development](#installation--local-development)
  - [Configuration](#configuration)
  - [Usage](#usage)
  - [Deployment](#deployment)
  - [Troubleshooting \& Tips](#troubleshooting--tips)
  - [Contributing](#contributing)
  - [License](#license)

## Features

- Noise reduction for any video format via DeepFilterNet3 AI model
- Secure direct-to-blob uploads/downloads with time-limited SAS tokens
- Serverless API (Azure Functions) and queue-based processing (Service Bus)
- GitHub OAuth authentication and Stripe subscription management
- Automatic cleanup of old files and cost-efficient scaling to zero

## Prerequisites

- Node.js (>=14.x) and npm
- Azure Static Web Apps CLI (`npm install -g @azure/static-web-apps-cli`)
- Azure Functions Core Tools (`npm install -g azure-functions-core-tools@4`)
- Docker (for local AI processor)
- Terraform (for infrastructure provisioning)

## Project Structure

```text
Audio-Cleaner-Web/
├── frontend/          # React + Vite UI
├── api/               # Azure Functions (Node.js)
│   ├── upload-file/   # Generate upload SAS URL
│   ├── enqueue-job/   # Enqueue processing job
│   ├── job-status/    # Poll job status
│   ├── download-file/ # Generate download SAS URL
│   └── shared/        # Utilities and middleware
├── processor/         # Python AI service container
│   ├── Dockerfile     # Container spec
│   └── src/           # DeepFilterNet3 inference code
└── terraform/         # Infrastructure as Code
```

## Architecture

```mermaid
graph LR
  F[Frontend (React)] -->|API calls| A[Azure Functions]
  A --> B[Service Bus]
  B --> C[AI Processor (Container App)]
  C --> D[Blob Storage]
  A --> E[Cosmos DB]
```

- Frontend & API scale to zero when idle
- Processor: containerized AI runs per job
- Storage & DB: secure, pay-per-use

## Installation & Local Development

1. Clone the repository:

   ```bash
   git clone https://github.com/BenjaminDanker/Audio-Cleaner-Web.git
   cd Audio-Cleaner-Web
   ```

2. Start frontend:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. Start API:

   ```bash
   cd ../api
   npm install
   npm run dev
   ```

4. Start AI processor:

   ```bash
   cd ../processor
   docker compose -f docker-compose.dev.yml up
   ```

5. Launch full stack locally:

   ```bash
   npx swa start
   ```

6. Open <http://localhost:4280> in your browser.

## Configuration

1. Copy and fill local settings:

   ```bash
   cd api
   cp local.settings.json.example local.settings.json
   ```

2. In `local.settings.json`, set:
   - `AzureWebJobsStorage` (Blob Storage connection string)
   - `SERVICE_BUS_CONNECTION` (Service Bus connection string)
   - `COSMOS_DB_CONNECTION` (Cosmos DB connection string)
   - `GITHUB_OAUTH_CLIENT_ID` / `GITHUB_OAUTH_CLIENT_SECRET`
   - `STRIPE_SECRET_KEY`

3. Terraform variables:

   Edit `terraform/terraform.tfvars` with your Azure subscription and resource group.

## Usage

1. Log in via GitHub on the frontend.
2. Upload a video → frontend requests a SAS URL → upload to Blob Storage.
3. Click **Process** → job enqueued on Service Bus.
4. View processing status in real time.
5. Download the processed video via SAS-protected URL.

## Deployment

1. Provision infrastructure:

   ```bash
   cd terraform
   terraform init
   terraform apply
   ```

2. Deploy frontend & API:

   ```bash
   npx swa deploy
   ```

3. Monitor services in the Azure portal as needed.

## Troubleshooting & Tips

- **Azure Functions errors**: run `func start --verbose` in the `api` directory.
- **Blob access issues**: verify SAS token validity and storage CORS settings.
- **Queue delays**: check Service Bus SKU and message metrics.
- **Processor rebuild**: rerun `docker build` if model files change.

## Contributing

1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Implement changes and add tests.
4. Test end-to-end locally.
5. Submit a pull request against `main`.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
