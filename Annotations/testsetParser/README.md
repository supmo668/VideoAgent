# Video Annotation Tool

This tool allows you to annotate video frames with structured information and push the annotations to Hugging Face datasets.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Update the values in `.env` with your credentials
```bash
cp .env.example .env
```

3. Login to Hugging Face:
```bash
huggingface-cli login
```

## Environment Variables

The following environment variables are required:

| Variable | Description | Required |
|----------|-------------|----------|
| `HUGGINGFACE_TOKEN` | Your Hugging Face API token | Yes |

To get your Hugging Face token:
1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with `write` access
3. Copy the token to your `.env` file

## Usage

1. Start the application:
```bash
reflex run
```

2. Access the web interface at `http://localhost:3000`

3. Features:
   - Load videos via URL
   - Add structured annotations with frame timestamps
   - Push annotations to Hugging Face datasets
   - Toggle dataset privacy settings

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
