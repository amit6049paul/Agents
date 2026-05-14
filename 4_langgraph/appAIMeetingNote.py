import gradio as gr
from huggingface_hub import InferenceClient


def transcribe_audio(audio_file_path):
    """Transcribe audio using an Inference Provider"""
    client = InferenceClient(provider="auto")

    # Pass the file path directly - the client handles file reading
    transcript = client.automatic_speech_recognition(
        audio=audio_file_path, model="openai/whisper-large-v3"
    )

    return transcript.text


def generate_summary(transcript):
    """Generate summary using an Inference Provider"""
    client = InferenceClient(provider="auto")

    prompt = f"""
    Analyze this meeting transcript and provide:
    1. A concise summary of key points
    2. Action items with responsible parties
    3. Important decisions made

    Transcript: {transcript}

    Format with clear sections:
    ## Summary
    ## Action Items
    ## Decisions Made
    """

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-0528:fastest",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )

    return response.choices[0].message.content


def process_meeting_audio(audio_file):
    """Main processing function"""
    if audio_file is None:
        return "Please upload an audio file.", ""

    try:
        # Step 1: Transcribe
        transcript = transcribe_audio(audio_file)

        # Step 2: Summarize
        summary = generate_summary(transcript)

        return transcript, summary

    except Exception as e:
        return f"Error processing audio: {str(e)}", ""


# Create Gradio interface
app = gr.Interface(
    fn=process_meeting_audio,
    inputs=gr.Audio(label="Upload Meeting Audio", type="filepath"),
    outputs=[
        gr.Textbox(label="Transcript", lines=10),
        gr.Textbox(label="Summary & Action Items", lines=8),
    ],
    title="🎤 AI Meeting Notes",
    description="Upload audio to get instant transcripts and summaries.",
)

if __name__ == "__main__":
    app.launch()