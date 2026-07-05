output "container_name" { value = docker_container.neo4j.name }
output "http_port" { value = var.http_port }
output "bolt_port" { value = var.bolt_port }
