# Application Load Balancer Security Group
resource "aws_security_group" "alb" {
  name        = "enterprise-agentic-alb-sg-${var.environment}"
  description = "Controls HTTP and HTTPS ingress to the ALB."
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "Allow HTTPS from Internet"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Allow HTTP from Internet (redirect to HTTPS)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "Allow all outbound traffic to worker nodes"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "enterprise-agentic-alb-sg-${var.environment}"
  }
}

# EKS Worker Nodes Security Group
resource "aws_security_group" "eks_nodes" {
  name        = "enterprise-agentic-eks-nodes-sg-${var.environment}"
  description = "Security group for EKS worker nodes and agent pods."
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Allow API traffic from ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    description = "Allow inter-node communication within EKS cluster"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
  }

  egress {
    description = "Allow all outbound traffic (NAT / external LLMs)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name                                        = "enterprise-agentic-eks-nodes-sg-${var.environment}"
    "kubernetes.io/cluster/${var.cluster_name}" = "owned"
  }
}

# Aurora / RDS PostgreSQL Security Group
resource "aws_security_group" "database" {
  name        = "enterprise-agentic-database-sg-${var.environment}"
  description = "Controls access to Aurora / RDS PostgreSQL pgvector instance."
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Allow PostgreSQL access strictly from EKS worker nodes"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }

  egress {
    description = "Allow outbound responses within VPC"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = [var.vpc_cidr]
  }

  tags = {
    Name = "enterprise-agentic-database-sg-${var.environment}"
  }
}

# ElastiCache Redis Security Group
resource "aws_security_group" "redis" {
  name        = "enterprise-agentic-redis-sg-${var.environment}"
  description = "Controls access to ElastiCache Redis cluster."
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Allow Redis access strictly from EKS worker nodes"
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes.id]
  }

  egress {
    description = "Allow outbound responses within VPC"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = [var.vpc_cidr]
  }

  tags = {
    Name = "enterprise-agentic-redis-sg-${var.environment}"
  }
}
