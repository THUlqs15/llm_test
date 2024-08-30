curl -X 'POST' \
  'http://127.0.0.1:8000/generate/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Hello, how are you?",
  "max_length": 100
}'
