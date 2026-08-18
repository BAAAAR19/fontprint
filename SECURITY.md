# Security policy

## Supported versions

The latest release on the default branch receives security fixes.

## Reporting a vulnerability

Do not attach sensitive documents, proprietary fonts, API logs, or exploit details to a public issue. Use GitHub's private vulnerability reporting for the repository. Include the affected version, minimal reproduction using synthetic data, impact, and suggested mitigation if known.

## Deployment notes

Fontprint's example API is a reference service, not a complete internet-facing security boundary. Production deployments should add authentication, TLS, rate limits, request timeouts, isolated image decoding, malware scanning, private logging defaults, and a documented deletion policy. Model output must not be used as an automatic fraud verdict.
