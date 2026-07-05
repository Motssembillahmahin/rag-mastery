output "network_name" {
  description = "Docker network name"
  value       = module.networking.network_name
}

output "chromadb_url" {
  description = "ChromaDB URL"
  value       = var.enable_chromadb ? "http://localhost:${var.chromadb_port}" : null
}

output "ollama_url" {
  description = "Ollama URL"
  value       = var.enable_ollama ? "http://localhost:${var.ollama_port}" : null
}

output "neo4j_url" {
  description = "Neo4j URL"
  value       = var.enable_neo4j ? "http://localhost:${var.neo4j_http_port}" : null
}

output "grafana_url" {
  description = "Grafana URL"
  value       = var.enable_monitoring ? "http://localhost:${var.grafana_port}" : null
}

output "prometheus_url" {
  description = "Prometheus URL"
  value       = var.enable_monitoring ? "http://localhost:${var.prometheus_port}" : null
}
