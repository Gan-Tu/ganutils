# API Root Endpoint
API_ROOT = "https://api.github.com"

# API Paths Endpoint
API_ENDPOINTS = {
  "issue": {
    "create": "/repos/{repo_owner}/{repo_name}/issues",
    "close": "/repos/{repo_owner}/{repo_name}/issues/{issue_number}"
  }
}

# API BUILDER
def get_api_endpoint(endpoint, method):
  if endpoint in API_ENDPOINTS:
    if method in API_ENDPOINTS[endpoint]:
      return API_ROOT + API_ENDPOINTS[endpoint][method]
    else:
      raise RuntimeError("Method not available: " + method)
  else:
    raise RuntimeError("Endpoint not available: " + endpoint)