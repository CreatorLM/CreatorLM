print("=== SCRIPT STARTED ===")
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import shutil
import os
import wave
import json
import requests
from typing import List
from dotenv import load_dotenv
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from datetime import datetime
import pytz
import statistics
import tempfile

# Import timezone data
from timezone_data import COUNTRY_TIMEZONES, YOUTUBE_UTC_TIMINGS, INSTAGRAM_UTC_TIMINGS

# Load API keys from .env
load_dotenv()
GOOGLE_AI_STUDIO_KEY = os.getenv("GOOGLE_AI_STUDIO_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Create the FastAPI app
app = FastAPI()

@app.get("/")
async def serve_homepage():
    return FileResponse("index.html")

@app.get("/sitemap.xml")
async def serve_sitemap():
    return FileResponse("sitemap.xml")

@app.get("/creatorlm-logo.png")
async def serve_logo():
    return FileResponse("creatorlm-logo.png")
    


# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],  # Add GET if needed
    allow_headers=["*"],
    expose_headers=["*"]  # ← Add this line
)

# Load Vosk model
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"  # Update this path to your Vosk model directory
vosk_model = Model(VOSK_MODEL_PATH)

# Add this new endpoint to serve your HTML UI
@app.get("/", response_class=HTMLResponse)
async def serve_html_ui():
    with open("index.html", "r") as file:
        return HTMLResponse(content=file.read())



# ====== Timezone Conversion Functions ======
def convert_utc_to_local(utc_time_str, user_tz):
    """Convert UTC time string to user's local timezone."""
    naive = datetime.strptime(utc_time_str, "%H:%M")
    utc_time = pytz.utc.localize(naive)
    return utc_time.astimezone(pytz.timezone(user_tz)).strftime("%H:%M")

def get_best_times(target_country, user_country, platform):
    """Get converted times for the platform."""
    try:
        user_tz = COUNTRY_TIMEZONES[user_country]
        user_day = datetime.now(pytz.timezone(user_tz)).strftime("%A")
        
        if target_country == "All (Global)":
            # Use global timings directly
            if platform == "youtube":
                timings = YOUTUBE_UTC_TIMINGS["Global"][user_day]
            else:
                timings = INSTAGRAM_UTC_TIMINGS["Global"][user_day]
            return [convert_utc_to_local(t, user_tz) for t in timings]
        
        # Use country-specific timings
        timings = YOUTUBE_UTC_TIMINGS[target_country][user_day] if platform == "youtube" else INSTAGRAM_UTC_TIMINGS[target_country][user_day]
        return [convert_utc_to_local(t, user_tz) for t in timings]
    except Exception as e:
        print(f"Error getting times: {str(e)}")
        return []

# ====== Original Functions (Unchanged) ======
def extract_audio_from_video(video_path, output_wav_path):
    """Extracts audio from a video file and converts it to WAV format using pydub."""
    try:
        print("Extracting audio from video...")
        audio = AudioSegment.from_file(video_path)
        audio.export(output_wav_path, format="wav")
        print("Audio extracted successfully.")
    except Exception as e:
        raise ValueError(f"Error extracting audio: {str(e)}")

def convert_to_vosk_format(input_wav_path, output_wav_path):
    """Converts the WAV file to the format required by Vosk (16kHz, mono, 16-bit)."""
    try:
        print("Converting audio to Vosk format...")
        audio = AudioSegment.from_file(input_wav_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_wav_path, format="wav")
        print("Audio converted successfully.")
    except Exception as e:
        raise ValueError(f"Error converting audio: {str(e)}")

def transcribe_audio_vosk(audio_path):
    """Uses Vosk for speech-to-text."""
    try:
        print("Transcribing audio using Vosk...")
        wf = wave.open(audio_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000]:
            raise ValueError("Audio file must be WAV format, mono, 16-bit, and 8kHz or 16kHz.")

        rec = KaldiRecognizer(vosk_model, wf.getframerate())
        result = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result += json.loads(rec.Result())["text"]
        print("Transcription complete.")
        return result
    except Exception as e:
        raise ValueError(f"Error transcribing audio: {str(e)}")

def fetch_youtube_metadata(query: str):
    """Fetches video metadata from YouTube API based on a search query."""
    try:
        print("Fetching YouTube metadata...")
        search_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&type=video&maxResults=5&key={YOUTUBE_API_KEY}"
        search_response = requests.get(search_url)
        search_response.raise_for_status()
        search_response_json = search_response.json()
        print("YouTube Search API Response:", json.dumps(search_response_json, indent=2))

        tags = []
        titles = []
        if "items" in search_response_json:
            for item in search_response_json["items"]:
                video_id = item["id"]["videoId"]
                video_url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&part=snippet&key={YOUTUBE_API_KEY}"
                print(f"Fetching metadata for video ID: {video_id}")
                video_response = requests.get(video_url)
                video_response.raise_for_status()
                video_response_json = video_response.json()
                print("YouTube Video API Response:", json.dumps(video_response_json, indent=2))

                if "items" in video_response_json and video_response_json["items"]:
                    snippet = video_response_json["items"][0]["snippet"]
                    tags.extend(snippet.get("tags", []))
                    titles.append(snippet["title"])

        # Deduplicate tags and limit to 25
        tags = list(set(tags))[:25]
        # Limit titles to 5
        titles = titles[:5]

        print("YouTube metadata fetched successfully.")
        return {"tags": tags, "titles": titles}
    except requests.exceptions.RequestException as e:
        print("YouTube API Error:", str(e))
        return {"tags": [], "titles": []}

def process_with_google_ai_studio(description: str, transcribed_text: str):
    """Uses Google AI Studio to generate titles and tags based on the description and transcribed text."""
    try:
        print("Starting Google AI Studio processing...")
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": GOOGLE_AI_STUDIO_KEY}

        # Prepare the request payload
        prompt = f"""
        You are a YouTube SEO expert. Generate 5 creative titles and 25 relevant tags for a YouTube video based on the following description and transcribed text:
        
        Description: {description}
        Transcribed Text: {transcribed_text}
        
        The video is about: "{description}".
        
        Return the output in the following JSON format:
        {{
          "titles": ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"],
          "tags": ["tag1", "tag2", "tag3", ...]
        }}
        """

        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }

        print("Sending request to Google AI Studio API...")
        print("Request Payload:", json.dumps(data, indent=2))
        response = requests.post(url, json=data, headers=headers, params=params)
        response.raise_for_status()
        print("Google AI Studio API Response:", json.dumps(response.json(), indent=2))

        # Extract the generated content
        generated_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        print("Generated Text:", generated_text)

        # Parse the generated text into titles and tags
        try:
            # Extract JSON from the generated text
            json_start = generated_text.find("{")
            json_end = generated_text.rfind("}") + 1
            json_str = generated_text[json_start:json_end]
            result = json.loads(json_str)

            # Extract titles and tags
            titles = result.get("titles", [])
            tags = result.get("tags", [])

            # Ensure titles and tags are lists
            if not isinstance(titles, list):
                titles = []
            if not isinstance(tags, list):
                tags = []

            print("Google AI Studio processing complete.")
            return {"titles": titles, "tags": tags}
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            print("Error parsing generated text:", str(e))
            return {"titles": [], "tags": []}
    except requests.exceptions.RequestException as e:
        print("Google AI Studio API Error:", str(e))
        return {"titles": [], "tags": []}

# ====== API Endpoints ======
@app.post("/process_video")
async def process_video(
    description: str = Form(...),
    target_country: str = Form(...),  # Must match frontend's "target_country"
    user_country: str = Form(...),    # Must match frontend's "user_country"
    file: UploadFile = File(...)      # Must be last parameter
):
    """Processes video, extracts keywords, fetches metadata, and suggests the best upload time."""
    try:
        print("\n--- Starting Video Processing ---")
        print("Received request with file:", file.filename)
        print("Description:", description)
        print("Target Country:", target_country)
        print("User Country:", user_country)

        # Save uploaded file to a temporary directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
            print(f"Saved uploaded file to temporary location: {temp_file_path}")

        # Extract audio from video
        wav_path = tempfile.mktemp(suffix=".wav")
        extract_audio_from_video(temp_file_path, wav_path)

        # Convert audio to Vosk format
        vosk_wav_path = tempfile.mktemp(suffix="_vosk.wav")
        convert_to_vosk_format(wav_path, vosk_wav_path)

        # Transcribe audio using Vosk
        extracted_text = transcribe_audio_vosk(vosk_wav_path)
        print("Extracted Text:", extracted_text)

        # Determine if video is a song or voiceover
        is_song = not any(word in extracted_text.lower() for word in description.lower().split())
        
        # Fetch YouTube metadata
        if is_song:
            print("Video is a song. Fetching metadata based on description...")
            metadata = fetch_youtube_metadata(description)
        else:
            print("Video is a voiceover. Fetching metadata based on transcribed text...")
            metadata = fetch_youtube_metadata(extracted_text)
        
        # Get best upload times
        youtube_times = get_best_times(target_country, user_country, "youtube")
        instagram_times = get_best_times(target_country, user_country, "instagram")

        # Process the description and transcribed text with Google AI Studio
        print("Starting Google AI Studio processing...")
        processed_data = process_with_google_ai_studio(description, extracted_text)

        response_data = {
            "status": "success",
            "tags": metadata["tags"] + processed_data["tags"],
            "titles": metadata["titles"] + processed_data["titles"],
            "youtube_times": youtube_times,
            "instagram_times": instagram_times,
            "message": "Processing complete!"
        }

        # Log the response
        print("\n--- Processing Complete ---")
        print("Response Data:", json.dumps(response_data, indent=2))

        # Delete temporary files
        print("\nCleaning up temporary files...")
        os.remove(temp_file_path)
        os.remove(wav_path)
        os.remove(vosk_wav_path)
        print("Temporary files deleted.")

        return response_data
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health_check/")
def health_check():
    """Checks if the API is running properly."""
    return {"status": "Running fine!"}
