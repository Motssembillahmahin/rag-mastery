resource "docker_container" "prometheus" {
  name  = "${var.project_name}-prometheus"
  image = "prom/prometheus:latest"
  ports {
    internal = 9090
    external = var.prometheus_port
  }
  networks_advanced {
    name = var.network_name
  }
  restart = "unless-stopped"
  labels {
    label = "service"
    value = "prometheus"
  }
}

resource "docker_container" "grafana" {
  name  = "${var.project_name}-grafana"
  image = "grafana/grafana:latest"
  ports {
    internal = 3000
    external = var.grafana_port
  }
  env = [
    "GF_SECURITY_ADMIN_PASSWORD=rag-mastery",
    "GF_USERS_ALLOW_SIGN_UP=false"
  ]
  networks_advanced {
    name = var.network_name
  }
  restart = "unless-stopped"
  labels {
    label = "service"
    value = "grafana"
  }
}

resource "docker_volume" "grafana_data" {
  name = "${var.project_name}-grafana-data"
}
