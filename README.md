<!-- markdownlint-disable MD031 MD032 MD040 MD022 MD036 MD058 MD026 MD009 MD024-->
# Audio Cleaner Pro

> AI-powered video denoising web application built on Azure

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/)

## ✨ Features

- 🎵 **AI Audio Denoising** - Remove background noise from videos using DeepFilterNet3
- 📁 **Drag & Drop Upload** - Intuitive file upload with progress tracking
- ⚡ **Real-time Processing** - Monitor job status and download results instantly
- 🔐 **Secure Authentication** - User accounts with subscription management
- 💳 **Stripe Integration** - Flexible payment plans
- 🌐 **Fully Scalable** - Azure-native architecture

## 🚀 Quick Start

### Prerequisites
- Azure subscription with Owner/Contributor access
- [Azure Developer CLI](https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/install-azd)
- [Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)

### Deploy to Azure

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Audio-Cleaner-Web
   ```

2. **Login and deploy**
   ```bash
   azd auth login
   azd up
   ```

That's it! The deployment will provision all Azure resources and deploy your application.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React SPA     │───▶│ Azure Functions │───▶│ Container Apps  │
│ (Static Web App)│    │    (Node.js)    │    │   (Python AI)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       ▼
         │              ┌─────────────────┐    ┌─────────────────┐
         │              │   Cosmos DB     │    │  Blob Storage   │
         │              │  (Metadata)     │    │ (Video Files)   │
         └─────────────▶└─────────────────┘    └─────────────────┘
```

**Components:**
- **Frontend**: React + Vite hosted on Azure Static Web Apps
- **API**: Node.js Azure Functions with JWT authentication
- **Processing**: Python container with DeepFilterNet3 AI model
- **Storage**: Cosmos DB for metadata, Blob Storage for files
- **Infrastructure**: Bicep templates with automated deployment

## � Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - Detailed system design and components
- **[Deployment Guide](docs/DEPLOYMENT.md)** - CI/CD setup and environment configuration
- **[Security Guide](docs/SECURITY.md)** - Authentication, authorization, and best practices
- **[Developer Guide](docs/DEVELOPMENT.md)** - Local setup and contribution guidelines

## 🔧 Configuration

Key environment variables (automatically configured during deployment):
- `AZURE_ENV_NAME` - Environment identifier
- `AZURE_LOCATION` - Deployment region
- `AZURE_SUBSCRIPTION_ID` - Target subscription

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
