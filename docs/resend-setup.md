# Resend Setup

1. Register at https://resend.com (free tier: 3,000 emails/month)
2. Create an API key at https://resend.com/api-keys
3. For production: verify your domain at https://resend.com/domains
   - Until domain is verified, you can only send to your own verified email
4. Set secrets on Fly.io:

```bash
fly secrets set \
  GOOGLE_API_KEY="your_gemini_api_key" \
  RESEND_API_KEY="re_your_resend_key" \
  ALLOWED_EMAILS="your@email.com,student1@email.com" \
  CORS_ORIGIN="https://zoom-to-text.fly.dev" \
  BASE_URL="https://zoom-to-text.fly.dev" \
  DATA_DIR="/data" \
  DOWNLOADS_DIR="/data/downloads"
```

5. First deploy:

```bash
fly auth login
fly launch --no-deploy --name zoom-to-text --region ams
fly volumes create zoom_data --region ams --size 10
fly deploy
fly logs
```
