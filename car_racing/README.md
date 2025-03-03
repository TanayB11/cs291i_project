# Reward Hacking Experiment

## Setup

### OpenAI API Configuration

To use the GPT-4o-mini evaluation feature, you need to set up your OpenAI API key:

1. **Option 1:** Set as environment variable:
   ```bash
   export OPENAI_API_KEY=your-api-key-here
   ```

2. **Option 2:** Edit `hacky3_config.py` and set your API key:
   ```python
   OPENAI_API_KEY = "your-api-key-here"
   ```

3. **Option 3:** Pass the API key directly when creating the callback:
   ```python
   callback = VideoRecordingCallback(
       save_freq=500,
       root_folder=root_folder,
       name_prefix='car_racing',
       api_key='your-api-key-here'
   )
   ```

## Running the Code

When running `hacky3.py`, it will generate:
- Checkpoints
- Grid images of agent behavior
- GPT evaluations (if API key is configured)
