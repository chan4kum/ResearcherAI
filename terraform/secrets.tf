resource "aws_secretsmanager_secret" "api_secrets" {
  name                    = "enterprise-agentic-platform-secrets-${var.environment}"
  description             = "Application secrets, API tokens, and database credentials for Enterprise Agentic Platform."
  kms_key_id              = aws_kms_key.platform_key.arn
  recovery_window_in_days = 7

  tags = {
    Name = "enterprise-agentic-secrets-${var.environment}"
  }
}
