resource "docker_network" "rag_network" {
  name = "${var.project_name}-${var.environment}-network"
  driver = "bridge"
  labels {
    label = "project"
    value = var.project_name
  }
  labels {
    label = "environment"
    value = var.environment
  }
}
