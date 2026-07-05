module "networking" {
  source = "./modules/networking"
  project_name = var.project_name
  environment  = var.environment
}

module "storage" {
  source = "./modules/storage"
  project_name = var.project_name
  environment  = var.environment
}

module "chromadb" {
  source = "./modules/chromadb"
  count  = var.enable_chromadb ? 1 : 0
  project_name = var.project_name
  network_name = module.networking.network_name
  data_volume  = module.storage.data_volume_name
  port         = var.chromadb_port
}

module "ollama" {
  source = "./modules/ollama"
  count  = var.enable_ollama ? 1 : 0
  project_name  = var.project_name
  network_name  = module.networking.network_name
  models_volume = module.storage.models_volume_name
  port          = var.ollama_port
  models        = var.ollama_models
  embedding_model = var.embedding_model
}

module "neo4j" {
  source = "./modules/neo4j"
  count  = var.enable_neo4j ? 1 : 0
  project_name   = var.project_name
  network_name   = module.networking.network_name
  data_volume    = module.storage.data_volume_name
  http_port      = var.neo4j_http_port
  bolt_port      = var.neo4j_bolt_port
  password       = "rag-mastery-password"
}

module "monitoring" {
  source = "./modules/monitoring"
  count  = var.enable_monitoring ? 1 : 0
  project_name   = var.project_name
  network_name   = module.networking.network_name
  prometheus_port = var.prometheus_port
  grafana_port   = var.grafana_port
}
