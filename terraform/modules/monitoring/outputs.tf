output "prometheus_container_name" { value = docker_container.prometheus.name }
output "grafana_container_name" { value = docker_container.grafana.name }
output "prometheus_port" { value = var.prometheus_port }
output "grafana_port" { value = var.grafana_port }
