##Experiment Pipeline build trigger
#resource "google_cloudbuild_trigger" "experiment-trigger" {
#  location = var.default_region
#  project = var.project_id
#  name = "clustering-experiment-pipeline-trigger"
#  filename = var.experiment-file
#  service_account = "projects/nashtech-ai-dev-389315/serviceAccounts/nashtech-ai-dev-app-sa@nashtech-ai-dev-389315.iam.gserviceaccount.com"
#  repository_event_config {
#    repository = google_cloudbuildv2_repository.my_repository.id
#    push {
#      branch = var.branch
#    }
#  }
#  approval_config {
#    approval_required = true
#  }
#}
#
##Main Pipeline Trigger
#resource "google_cloudbuild_trigger" "pipeline-trigger" {
#  location = var.default_region
#
#  project = var.project_id
#  name = "clustering-main-pipeline-trigger"
#  filename = var.main-file
#  service_account = "projects/nashtech-ai-dev-389315/serviceAccounts/nashtech-ai-dev-app-sa@nashtech-ai-dev-389315.iam.gserviceaccount.com"
#
#  repository_event_config {
#    repository = google_cloudbuildv2_repository.my_repository.id
#    push {
#      branch = var.branch
#    }
#  }
##  approval_config {
##    approval_required = true
##  }
#}
#
#
#
#Serve Trigger
#resource "google_cloudbuild_trigger" "serve-trigger" {
#  location = var.default_region
#
#  project = var.project_id
#  name = "clustering-serve-trigger"
#  filename = var.serve-file
#  service_account = var.service_account
#
#  repository_event_config {
#    repository = google_cloudbuildv2_repository.my_repository.id
#  }
#}

#Pub-Sub Topic'''
resource "google_pubsub_topic" "serve-trigger-topic" {
  name = var.serve_trigger_pub_sub_topic
}

# Pub-Sub Serve Trigger
resource "google_cloudbuild_trigger" "serve-trigger" {
  location = var.default_region
  project = var.project_id
  name = "clustering-serve-trigger"
  service_account = var.service_account

  pubsub_config {
    topic = google_pubsub_topic.serve-trigger-topic.id
    service_account_email = var.service_account
  }

  source_to_build {
    repository = google_cloudbuildv2_repository.my_repository.id
    ref = "refs/heads/main"
    repo_type = "GITHUB"
  }

  git_file_source {
    path = var.serve-file
    repository = google_cloudbuildv2_repository.my_repository.id
    revision = "refs/heads/main"
    repo_type = "GITHUB"
  }

}


