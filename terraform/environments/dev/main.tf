module "rag_infrastructure" {
  source = "../../"
  project_name = "rag-mastery"
  environment  = "dev"
  enable_chromadb   = true
  enable_ollama     = true
  enable_neo4j      = false
  enable_monitoring = false
  chromadb_port    = 8000
  ollama_port      = 11434
  prometheus_port  = 9090
  grafana_port     = 3000
  ollama_models    = ["llama3", "mistral"]
  embedding_model  = "all-minilm:latest"
}
