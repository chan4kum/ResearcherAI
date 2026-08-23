variable "aws_region" {
  type        = string
  description = "AWS deployment region."
  default     = "us-east-1"
}

variable "environment" {
  type        = string
  description = "Target deployment environment (development, staging, production)."
  default     = "production"
}

variable "vpc_cidr" {
  type        = string
  description = "Primary CIDR block for the VPC."
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  type        = list(string)
  description = "Availability zones for multi-AZ topology."
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "public_subnet_cidrs" {
  type        = list(string)
  description = "CIDR blocks for public subnets (ALBs & NATs)."
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_app_subnet_cidrs" {
  type        = list(string)
  description = "CIDR blocks for private application subnets (EKS worker nodes & pods)."
  default     = ["10.0.10.0/24", "10.0.20.0/24", "10.0.30.0/24"]
}

variable "isolated_data_subnet_cidrs" {
  type        = list(string)
  description = "CIDR blocks for isolated data subnets (Aurora PostgreSQL & ElastiCache Redis)."
  default     = ["10.0.100.0/24", "10.0.101.0/24", "10.0.102.0/24"]
}

variable "cluster_name" {
  type        = string
  description = "EKS Cluster identifier."
  default     = "enterprise-agentic-eks"
}

variable "enable_nat_gateway" {
  type        = bool
  description = "Provision NAT Gateways for private app subnet outbound connectivity."
  default     = true
}

variable "single_nat_gateway" {
  type        = bool
  description = "Use a single NAT Gateway across AZs to minimize non-production cost."
  default     = true
}
