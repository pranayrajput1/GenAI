variable "project_service_account" {
  type    = string
  default = "nashtech-ai-dev-app-sa@nashtech-ai-dev-389315.iam.gserviceaccount.com"
}

variable "service_account" {
  type = string
  default = "projects/nashtech-ai-dev-389315/serviceAccounts/nashtech-ai-dev-app-sa@nashtech-ai-dev-389315.iam.gserviceaccount.com"
}


variable "default_region" {
  description = "Default region for resources."
  type        = string
  default = "us-central1"
}

variable "project_number" {
  description = "Default project for resources."
  type        = string
  default = "24562761082"
}


variable "project_id" {
  description = "Project ID"
  type        = string
  default = "nashtech-ai-dev-389315"
}

variable "owner" {
  description = "Owner"
  type        = string
  default = "amanknoldus"
}

variable "repository" {
  description = "Repository"
  type        = string
  default = "test-pipeline"
}

variable "repository_uri" {
  description = "Repository URI"
  type        = string
  default = "https://github.com/amanknoldus/test-pipeline.git"
}

variable "branch" {
  description = "Branch"
  type        = string
  default = "main"
}

variable "experiment-file" {
  description = "Experiment-File"
  type        = string
  default = "experiment_cloudbuild.yaml"
}

variable "main-file" {
  description = "Pipeline-File"
  type        = string
  default = "cloudbuild.yaml"
}

variable "serve-file" {
  description = "Serve-File"
  type        = string
  default = "serving_container/serve_model_build.yaml"
}


variable "git_token" {
  type        = string
  default     = "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"
}

variable "serve_trigger_pub_sub_topic" {
  type = string
  default = "serve-model-topic"
}