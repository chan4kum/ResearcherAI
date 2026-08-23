output "vpc_id" {
  description = "The ID of the primary VPC."
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets."
  value       = aws_subnet.public[*].id
}

output "private_app_subnet_ids" {
  description = "IDs of the private application subnets (EKS)."
  value       = aws_subnet.private_app[*].id
}

output "isolated_data_subnet_ids" {
  description = "IDs of the isolated data subnets (Aurora & Redis)."
  value       = aws_subnet.isolated_data[*].id
}

output "s3_gateway_endpoint_id" {
  description = "The ID of the S3 Gateway VPC Endpoint."
  value       = aws_vpc_endpoint.s3.id
}

output "security_group_alb_id" {
  description = "Security Group ID for the Application Load Balancer."
  value       = aws_security_group.alb.id
}

output "security_group_eks_nodes_id" {
  description = "Security Group ID for the EKS worker nodes."
  value       = aws_security_group.eks_nodes.id
}

output "security_group_database_id" {
  description = "Security Group ID for Aurora / RDS PostgreSQL."
  value       = aws_security_group.database.id
}

output "security_group_redis_id" {
  description = "Security Group ID for ElastiCache Redis."
  value       = aws_security_group.redis.id
}

output "kms_key_arn" {
  description = "ARN of the Customer Managed KMS Key."
  value       = aws_kms_key.platform_key.arn
}

output "s3_knowledge_lake_bucket_name" {
  description = "Name of the S3 Knowledge Document Lake bucket."
  value       = aws_s3_bucket.knowledge_lake.id
}

output "ecr_repository_url" {
  description = "URL of the Amazon ECR repository."
  value       = aws_ecr_repository.api.repository_url
}

output "secrets_manager_secret_arn" {
  description = "ARN of the Secrets Manager secret."
  value       = aws_secretsmanager_secret.api_secrets.arn
}
