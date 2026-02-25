#!/bin/bash
# ALBuMS Docker Complete Image Build Script
# Includes mbtrack2 and pycolleff

set -e

echo "========================================"
echo "ALBuMS Docker Complete Image Build Script"
echo "========================================"
echo ""

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check required directories
echo -e "${BLUE}Step 1: Checking dependencies${NC}"

if [ ! -d "mbtrack2-stable" ]; then
    echo -e "${RED}Error: mbtrack2-stable directory not found${NC}"
    echo "Please ensure the mbtrack2-stable directory exists in the current directory."
    exit 1
fi

if [ ! -d "collective_effects" ]; then
    echo -e "${RED}Error: collective_effects directory not found${NC}"
    echo "Please ensure the collective_effects directory exists in the current directory."
    exit 1
fi

echo -e "${GREEN}✓ All dependency directories found${NC}"
echo ""

# Select Dockerfile
echo -e "${BLUE}Step 2: Selecting build method${NC}"
echo "1) Use local dependencies (Dockerfile.local) - Recommended"
echo "2) Download dependencies from network (Dockerfile)"
read -p "Select (1 or 2, default 1): " choice
choice=${choice:-1}

if [ "$choice" = "1" ]; then
    DOCKERFILE="Dockerfile.local"
    echo -e "${GREEN}Building with local dependencies${NC}"
else
    DOCKERFILE="Dockerfile"
    echo -e "${YELLOW}Building with network download (may require access permissions)${NC}"
fi
echo ""

# Build mirror
echo -e "${BLUE}Step 3: Building Docker image${NC}"
IMAGE_NAME="albums-streamlit"
IMAGE_TAG="latest"

echo "Building image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "This may take a few minutes..."
echo ""

if docker build -f "$DOCKERFILE" -t "${IMAGE_NAME}:${IMAGE_TAG}" .; then
    echo ""
    echo -e "${GREEN}✓ Docker image built successfully!${NC}"
else
    echo ""
    echo -e "${RED}✗ Docker image build failed${NC}"
    exit 1
fi
echo ""

# Display image info
echo -e "${BLUE}Step 4: Image Information${NC}"
docker images "${IMAGE_NAME}:${IMAGE_TAG}"
echo ""

# Ask whether to export
read -p "Export image to a tar file for sharing? (y/n): " export_choice
if [ "$export_choice" = "y" ]; then
    echo ""
    echo -e "${BLUE}Step 5: Exporting image${NC}"
    
    # Generate filename (with date)
    DATE=$(date +%Y%m%d)
    OUTPUT_FILE="albums-streamlit-complete-${DATE}.tar"
    
    echo "Exporting image to: ${OUTPUT_FILE}"
    echo "This may take a few minutes..."
    
    if docker save -o "${OUTPUT_FILE}" "${IMAGE_NAME}:${IMAGE_TAG}"; then
        echo ""
        echo -e "${GREEN}✓ Image exported successfully${NC}"
        
        # Display file size
        FILE_SIZE=$(du -h "${OUTPUT_FILE}" | cut -f1)
        echo "File: ${OUTPUT_FILE}"
        echo "Size: ${FILE_SIZE}"
        echo ""
        
        # Compress image
        read -p "Compress the image file? (y/n): " compress_choice
        if [ "$compress_choice" = "y" ]; then
            echo ""
            echo "Compressing..."
            if gzip "${OUTPUT_FILE}"; then
                COMPRESSED_FILE="${OUTPUT_FILE}.gz"
                COMPRESSED_SIZE=$(du -h "${COMPRESSED_FILE}" | cut -f1)
                echo -e "${GREEN}✓ Image compressed successfully${NC}"
                echo "Compressed File: ${COMPRESSED_FILE}"
                echo "Compressed Size: ${COMPRESSED_SIZE}"
            else
                echo -e "${RED}Compression failed${NC}"
            fi
        fi
    else
        echo -e "${RED}✗ Image export failed${NC}"
    fi
fi

echo ""
echo "========================================"
echo -e "${GREEN}Build Complete!${NC}"
echo "========================================"
echo ""
echo "Usage:"
echo ""
echo "1. Run locally:"
echo "   docker run -p 8501:8501 ${IMAGE_NAME}:${IMAGE_TAG}"
echo ""
echo "2. Using docker-compose:"
echo "   docker-compose up"
echo ""
echo "3. Sharing with others:"
if [ -f "albums-streamlit-complete-${DATE}.tar.gz" ]; then
    echo "   Share file: albums-streamlit-complete-${DATE}.tar.gz"
    echo "   Recipient runs: gunzip albums-streamlit-complete-${DATE}.tar.gz"
    echo "                  docker load -i albums-streamlit-complete-${DATE}.tar"
    echo "                  docker run -p 8501:8501 ${IMAGE_NAME}:${IMAGE_TAG}"
elif [ -f "albums-streamlit-complete-${DATE}.tar" ]; then
    echo "   Share file: albums-streamlit-complete-${DATE}.tar"
    echo "   Recipient runs: docker load -i albums-streamlit-complete-${DATE}.tar"
    echo "                  docker run -p 8501:8501 ${IMAGE_NAME}:${IMAGE_TAG}"
fi
echo ""
echo "Access application: http://localhost:8501"
echo ""
