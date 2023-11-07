resource "google_secret_manager_secret" "github_token_secret" {
    project =  var.project_id
    secret_id = "pipeline-trial"

      replication {
          user_managed {
          replicas {
          location = "us-central1"
        }
      }
    }
}

resource "google_secret_manager_secret_version" "github_token_secret_version" {
    secret = google_secret_manager_secret.github_token_secret.id
    secret_data = var.git_token
}

resource "google_secret_manager_secret_iam_policy" "policy" {
  project = google_secret_manager_secret.github_token_secret.project
  secret_id = google_secret_manager_secret.github_token_secret.secret_id
  policy_data = data.google_iam_policy.serviceagent_secretAccessor.policy_data
}

data "google_iam_policy" "serviceagent_secretAccessor" {
    binding {
        role = "roles/secretmanager.secretAccessor"
        members = [
          "serviceAccount:service-${var.project_number}@gcp-sa-cloudbuild.iam.gserviceaccount.com"
        ]
    }
}


// Create the GitHub connection
resource "google_cloudbuildv2_connection" "my_connection" {
    project = var.project_id
    location = "us-central1"
    name = "test-pipeline"
    github_config {
        app_installation_id = 40534709
        authorizer_credential {
            oauth_token_secret_version = google_secret_manager_secret_version.github_token_secret_version.id
        }
    }
   depends_on = [google_secret_manager_secret_iam_policy.policy]
}

resource "google_cloudbuildv2_repository" "my_repository" {
      project = var.project_id
      location = var.default_region
      name = var.repository
      parent_connection = google_cloudbuildv2_connection.my_connection.name
      remote_uri = var.repository_uri
  }