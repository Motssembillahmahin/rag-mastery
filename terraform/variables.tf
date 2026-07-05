variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "rag-mastery"
}

variable "environment" {
  description = "Environment name (dev/prod)"
  type        = string
  default     = "dev"
}

variable "enable_chromadb" {
  description = "Enable ChromaDB service"
  type        = bool
  default     = true
}

variable "enable_ollama" {
  description = "Enable Ollama service"
  type        = bool
  default     = true
}

variable "enable_neo4j" {
  description = "Enable Neo4j service"
  type        = bool
  default     = false
}

variable "enable_monitoring" {
  description = "Enable Prometheus + Grafana"
  type        = bool
  default     = false
}

variable "chromadb_port" {
  description = "ChromaDB external port"
  type        = number
  default     = 8000
}

variable "ollama_port" {
  description = "Ollama external port"
  type        = number
  default     = 11434
}

variable "neo4j_http_port" {
  description = "Neo4j HTTP port"
  type        = number
  default     = 7474
}

variable "neo4j_bolt_port" {
  description = "Neo4j Bolt port"
  type        = number
  default     = 7687
}

variable "prometheus_port" {
  description = "Prometheus port"
  type        = number
  default     = 9090
}

variable "grafana_port" {
  description = "Grafana port"
  type        = number
  default     = 3000
}

variable "ollama_models" {
  description = "Models to pull in Ollama"
  type        = list(string)
  default     = ["llama3", "mistral"]
}

variable "embedding_model" {
  description = "Default embedding model"
  type        = string
  default     = "all-minilm:latest"
}
