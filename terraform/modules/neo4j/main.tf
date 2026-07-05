resource "docker_container" "neo4j" {
  name  = "${var.project_name}-neo4j"
  image = "neo4j:latest"
  ports {
    internal = 7474
    external = var.http_port
  }
  ports {
    internal = 7687
    external = var.bolt_port
  }
  volumes {
    container_path = "/data"
    volume_name    = var.data_volume
  }
  env = [
    "NEO4J_AUTH=neo4j/${var.password}",
    "NEO4J_PLUGINS=[\"apoc\", \"graph-data-science\"]"
  ]
  networks_advanced {
    name = var.network_name
  }
  restart = "unless-stopped"
  labels {
    label = "service"
    value = "neo4j"
  }
}
