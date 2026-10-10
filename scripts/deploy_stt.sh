#!/usr/bin/env bash
# Triển khai stt-service lên Google Cloud Run (D119). Chạy tay bởi người có quyền project.
#
#   PROJECT=veggielife-510817 REGION=asia-southeast1 ./scripts/deploy_stt.sh
#
# Điều kiện (làm một lần):
#   gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
#   gcloud artifacts repositories create veggielife --repository-format=docker --location="$REGION"
#   gcloud iam service-accounts create stt-service
#   gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:stt-service@$PROJECT.iam.gserviceaccount.com" --role=roles/aiplatform.user
#   printf '%s' "<INTERNAL_API_KEY>" | gcloud secrets create stt-internal-api-key --data-file=-
#   gcloud secrets add-iam-policy-binding stt-internal-api-key --member="serviceAccount:stt-service@$PROJECT.iam.gserviceaccount.com" --role=roles/secretmanager.secretAccessor
set -euo pipefail
: "${PROJECT:?đặt PROJECT}"; REGION="${REGION:-asia-southeast1}"; WHISPER_MODEL="${WHISPER_MODEL:-small}"
MEMORY="${MEMORY:-4Gi}"; CPU="${CPU:-2}"; MIN_INSTANCES="${MIN_INSTANCES:-0}"
IMAGE="$REGION-docker.pkg.dev/$PROJECT/veggielife/stt-service:$(git rev-parse --short HEAD)"

gcloud builds submit --project "$PROJECT" --region "$REGION" --config cloudbuild.stt.yaml \
  --substitutions "_IMAGE=$IMAGE,_MODEL=$WHISPER_MODEL" .

gcloud run deploy stt-service --project "$PROJECT" --region "$REGION" --image "$IMAGE" \
  --service-account "stt-service@$PROJECT.iam.gserviceaccount.com" \
  --memory "$MEMORY" --cpu "$CPU" --min-instances "$MIN_INSTANCES" --max-instances 3 \
  --concurrency 1 --timeout 600 --cpu-boost \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=true,GOOGLE_CLOUD_PROJECT=$PROJECT,GOOGLE_CLOUD_LOCATION=$REGION,WHISPER_MODEL=$WHISPER_MODEL,WHISPER_COMPUTE_TYPE=int8,WHISPER_DEVICE=cpu,WHISPER_CPU_THREADS=$CPU" \
  --set-secrets "INTERNAL_API_KEY=stt-internal-api-key:latest"

echo "Xong. URL (đặt vào SttService__BaseUrl của Backend):"
gcloud run services describe stt-service --project "$PROJECT" --region "$REGION" --format='value(status.url)'
