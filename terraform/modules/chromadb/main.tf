resource "docker_container" "chromadb" {
  name  = "${var.project_name}-chromadb"
  image = "chromadb/chroma:latest"
  ports {
    internal = 8000
    external = var.port
  }
  volumes {
    container_path = "/chroma/chroma"
    volume_name    = var.data_volume
  }
  env = [
    "ANONYMIZED_TELEMETRY=False",
    "IS_PERSISTENT=TRUE",
    "PERSIST_DIRECTORY=/chroma/chroma"
  ]
  networks_advanced {
    name = var.network_name
  }
  restart = "unless-stopped"
  labels {
    label = "service"
    value = "chromadb"
  }
}
