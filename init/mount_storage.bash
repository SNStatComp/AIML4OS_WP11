cat <<EOF
[s3]
env_auth = false
type = s3
access_key_id = $AWS_ACCESS_KEY_ID
secret_access_key = $AWS_SECRET_ACCESS_KEY
region = $AWS_DEFAULT_REGION
endpoint = $AWS_S3_ENDPOINT
EOF
