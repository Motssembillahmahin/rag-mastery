resource "docker_container" "ollama" {
  name  = "${var.project_name}-ollama"
  image = "ollama/ollama:latest"
  ports {
    internal = 11434
    external = var.port
  }
  volumes {
    container_path = "/root/.ollama"
    volume_name    = var.models_volume
  }
  networks_advanced {
    name = var.network_name
  }
  restart = "unless-stopped"
  labels {
    label = "service"
    value = "ollama"
  }
  provisioner "local-exec" {
    command = <<-EOT
      sleep 10
      docker exec ${self.name} ollama pull ${var.embedding_model}
      ${join("\n      ", [for m in var.models : "docker exec ${self.name} ollama pull ${m}"])}
    EOT
  }
}
