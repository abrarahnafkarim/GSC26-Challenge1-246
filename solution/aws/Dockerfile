# Image for the AWS Batch sweep. Build, tag, and push to ECR:
#   aws ecr get-login-password --region REGION | docker login --username AWS \
#       --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com
#   docker build -t gsc2026 -f solution/aws/Dockerfile .
#   docker tag gsc2026:latest ACCOUNT.dkr.ecr.REGION.amazonaws.com/gsc2026:latest
#   docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/gsc2026:latest
#
# For CPU-only Batch, swap the base image for python:3.11-slim and drop the
# GPU resourceRequirement in batch_job_definition.json.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app
RUN pip install --no-cache-dir awscli torchvision numpy

# Bake the challenge repo into the image (run docker build from repo root).
COPY . /app

ENV PYTHONUNBUFFERED=1
CMD ["python", "solution/sweep.py", "--data-root", "/data/celeba", "--case", "0"]
