# Audio Cleaner Pro - Local Development Scripts

## Start Development Environment

### Frontend (Vite React)
```bash
cd frontend
npm install
npm run dev
```
The frontend will be available at http://localhost:5173

### Azure Functions API
```bash
cd api
npm install
func start --cors
```
The API will be available at http://localhost:7071

### Python Processing Service (Local)
```bash
# Install dependencies
pip install -r requirements.txt

# Run the processor
python processor_main.py
```

## Environment Configuration

### Frontend (.env.local)
Create `frontend/.env.local`:
```
VITE_API_BASE_URL=http://localhost:7071
```

### Azure Functions (local.settings.json)
Create `api/local.settings.json`:
```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "node",
    "COSMOS_CONNECTION_STRING": "AccountEndpoint=https://localhost:8081/;AccountKey=C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==",
    "SERVICE_BUS_CONNECTION_STRING": "Endpoint=sb://localhost:10000/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=AAAAAAAAAAAAAAAAAAAAAA==",
    "STRIPE_SECRET_KEY": "sk_test_your_test_key_here",
    "STRIPE_WEBHOOK_SECRET": "whsec_your_webhook_secret_here"
  }
}
```

### Python Service (.env)
Create `.env` in the root directory:
```
AZURE_STORAGE_CONNECTION_STRING=UseDevelopmentStorage=true
AZURE_SERVICE_BUS_CONNECTION_STRING=Endpoint=sb://localhost:10000/;SharedAccessKeyName=RootManageSharedAccessKey;SharedAccessKey=AAAAAAAAAAAAAAAAAAAAAA==
COSMOS_CONNECTION_STRING=AccountEndpoint=https://localhost:8081/;AccountKey=C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==
```

## Local Emulators

### Azure Storage Emulator (Azurite)
```bash
npm install -g azurite
azurite --silent --location c:\azurite --debug c:\azurite\debug.log
```

### Cosmos DB Emulator
Download and install the Azure Cosmos DB Emulator from Microsoft's website.

### Service Bus Emulator
For local development, you can use the Azure Service Bus Explorer or mock the queue with a simple in-memory queue.

## Testing the Complete Flow

1. Start all services (frontend, API, processor)
2. Navigate to http://localhost:5173
3. Sign in (will redirect to /.auth/me - in local development this may need to be mocked)
4. Upload an audio file
5. Monitor the processing queue and job status

## Debugging

### Frontend
- Use browser developer tools
- Check console for errors
- Verify API calls in Network tab

### Azure Functions
- Check the func host output
- Use VS Code Azure Functions extension for debugging
- Test individual endpoints with Postman or curl

### Python Processor
- Check console output for processing logs
- Use VS Code Python debugging
- Monitor temp files and model output

## Production Deployment Notes

When deploying to production:
1. Use real Azure resources (Storage, Cosmos DB, Service Bus)
2. Configure proper authentication (Managed Identity)
3. Set up proper secrets in Azure Key Vault
4. Use Azure Static Web Apps for frontend hosting
5. Deploy functions to Azure Functions
6. Deploy processor to Azure Container Instances or Container Apps
