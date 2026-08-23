# Primary VPC
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name                                        = "enterprise-agentic-vpc-${var.environment}"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# Internet Gateway
resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "enterprise-agentic-igw-${var.environment}"
  }
}

# Public Subnets (ALBs & NAT Gateways)
resource "aws_subnet" "public" {
  count                   = length(var.public_subnet_cidrs)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name                                        = "enterprise-agentic-public-${var.availability_zones[count.index]}-${var.environment}"
    "kubernetes.io/role/elb"                    = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    Tier                                        = "public"
  }
}

# Private Application Subnets (EKS Worker Nodes & Pods)
resource "aws_subnet" "private_app" {
  count             = length(var.private_app_subnet_cidrs)
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_app_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name                                        = "enterprise-agentic-private-app-${var.availability_zones[count.index]}-${var.environment}"
    "kubernetes.io/role/internal-elb"           = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
    Tier                                        = "private-app"
  }
}

# Isolated Data Subnets (Aurora PostgreSQL & ElastiCache Redis)
resource "aws_subnet" "isolated_data" {
  count             = length(var.isolated_data_subnet_cidrs)
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.isolated_data_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name = "enterprise-agentic-isolated-data-${var.availability_zones[count.index]}-${var.environment}"
    Tier = "isolated-data"
  }
}

# Elastic IP for NAT Gateway
resource "aws_eip" "nat" {
  count  = var.enable_nat_gateway ? (var.single_nat_gateway ? 1 : length(var.public_subnet_cidrs)) : 0
  domain = "vpc"

  tags = {
    Name = "enterprise-agentic-nat-eip-${count.index + 1}-${var.environment}"
  }
}

# NAT Gateway
resource "aws_nat_gateway" "nat" {
  count         = var.enable_nat_gateway ? (var.single_nat_gateway ? 1 : length(var.public_subnet_cidrs)) : 0
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = {
    Name = "enterprise-agentic-nat-${count.index + 1}-${var.environment}"
  }

  depends_on = [aws_internet_gateway.gw]
}

# Public Route Table
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.gw.id
  }

  tags = {
    Name = "enterprise-agentic-public-rt-${var.environment}"
  }
}

# Public Route Table Associations
resource "aws_route_table_association" "public" {
  count          = length(var.public_subnet_cidrs)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# Private Application Route Tables
resource "aws_route_table" "private_app" {
  count  = var.single_nat_gateway ? 1 : length(var.private_app_subnet_cidrs)
  vpc_id = aws_vpc.main.id

  dynamic "route" {
    for_each = var.enable_nat_gateway ? [1] : []
    content {
      cidr_block     = "0.0.0.0/0"
      nat_gateway_id = aws_nat_gateway.nat[var.single_nat_gateway ? 0 : count.index].id
    }
  }

  tags = {
    Name = "enterprise-agentic-private-app-rt-${count.index + 1}-${var.environment}"
  }
}

# Private Application Route Table Associations
resource "aws_route_table_association" "private_app" {
  count          = length(var.private_app_subnet_cidrs)
  subnet_id      = aws_subnet.private_app[count.index].id
  route_table_id = aws_route_table.private_app[var.single_nat_gateway ? 0 : count.index].id
}

# Isolated Data Route Table (No internet route)
resource "aws_route_table" "isolated_data" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "enterprise-agentic-isolated-data-rt-${var.environment}"
  }
}

# Isolated Data Route Table Associations
resource "aws_route_table_association" "isolated_data" {
  count          = length(var.isolated_data_subnet_cidrs)
  subnet_id      = aws_subnet.isolated_data[count.index].id
  route_table_id = aws_route_table.isolated_data.id
}

# S3 Gateway Endpoint (Free intra-VPC S3 access)
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.main.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids = concat(
    [aws_route_table.public.id],
    aws_route_table.private_app[*].id,
    [aws_route_table.isolated_data.id]
  )

  tags = {
    Name = "enterprise-agentic-s3-endpoint-${var.environment}"
  }
}
