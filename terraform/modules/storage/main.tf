resource "docker_volume" "data_volume" {
  name = "${var.project_name}-${var.environment}-data"
  labels {
    label = "project"
    value = var.project_name
  }
}

resource "docker_volume" "models_volume" {
  name = "${var.project_name}-${var.environment}-models"
  labels {
    label = "project"
    value = var.project_name
  }
}
