variable "project_name" { type = string }
variable "network_name" { type = string }
variable "data_volume" { type = string }
variable "http_port" { type = number }
variable "bolt_port" { type = number }
variable "password" { type = string; sensitive = true }
