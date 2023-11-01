# from google.cloud import secretmanager
#
# client = secretmanager.SecretManagerServiceClient()
# secret_name = "projects/nashtech-ai-dev-389315/secrets/access-token/versions/1"
# response = client.access_secret_version(name=secret_name)
# secret_value = response.payload.data.decode("UTF-8")
# print(f"Secret Value: {secret_value}")
