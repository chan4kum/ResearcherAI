terraform {
  required_version = ">= 1.5.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  # Production S3 Backend Configuration Placeholder
  # backend "s3" {
  #   bucket         = "enterprise-agentic-terraform-state"
  #   key            = "platform/prod/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "enterprise-agentic-terraform-locks"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  # Allows offline/CI plan validation without live cloud API dependencies
  skip_credentials_validation = true
  skip_requesting_account_id  = true
  skip_metadata_api_check     = true

  default_tags {
    tags = {
      Project     = "enterprise-agentic-platform"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

