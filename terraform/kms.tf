resource "aws_kms_key" "platform_key" {
  description             = "Customer Managed Key (CMK) for Enterprise Agentic Platform envelope encryption."
  deletion_window_in_days = 30
  enable_key_rotation     = true

  tags = {
    Name = "enterprise-agentic-kms-${var.environment}"
  }
}

resource "aws_kms_alias" "platform_key_alias" {
  name          = "alias/enterprise-agentic-platform-${var.environment}"
  target_key_id = aws_kms_key.platform_key.key_id
}
