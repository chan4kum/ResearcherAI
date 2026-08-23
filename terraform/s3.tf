resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket" "knowledge_lake" {
  bucket        = "enterprise-agentic-knowledge-lake-${var.environment}-${random_id.bucket_suffix.hex}"
  force_destroy = false

  tags = {
    Name = "enterprise-agentic-knowledge-lake-${var.environment}"
  }
}

resource "aws_s3_bucket_versioning" "knowledge_lake" {
  bucket = aws_s3_bucket.knowledge_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "knowledge_lake" {
  bucket = aws_s3_bucket.knowledge_lake.id

  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.platform_key.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "knowledge_lake" {
  bucket = aws_s3_bucket.knowledge_lake.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
