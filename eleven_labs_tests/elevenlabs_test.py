import os
import tempfile
import time
import requests
from typing import Optional
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

class SpeechToText:
    def __init__(self):
        """Initialize the STT module using the API key from .env"""
        self.api_key = ELEVENLABS_API_KEY
        self.temp_dir = tempfile.mkdtemp()
        print(f"Temporary directory created at: {self.temp_dir}")
    
    def record_audio_using_sounddevice(self, duration: int = 5, filename: Optional[str] = None) -> str:
        """
        Record audio from microphone using sounddevice
        
        Args:
            duration: Recording duration in seconds
            filename: Optional filename, if None a temporary file will be created
            
        Returns:
            Path to the recorded audio file
        """
        if filename is None:
            # Create temporary file
            audio_file = os.path.join(self.temp_dir, f"recording_{int(time.time())}.wav")
        else:
            audio_file = filename
        
        print(f"Recording audio for {duration} seconds using sounddevice...")
        
        # Audio parameters
        sample_rate = 16000  # 16 kHz
        channels = 1  # Mono
        
        try:
            # Record audio
            print("Recording started... Speak now")
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype='float32'
            )
            
            # Wait until recording is done
            sd.wait()
            print("Recording finished")
            
            # Save as WAV file
            sf.write(audio_file, recording, sample_rate)
            print(f"Audio saved to {audio_file}")
            
            return audio_file
        
        except Exception as e:
            print(f"Error recording audio with sounddevice: {e}")
            return None
    
    def list_audio_devices(self):
        """List available audio input devices using sounddevice"""
        try:
            devices = sd.query_devices()
            print("\n=== Available Audio Devices ===")
            for i, device in enumerate(devices):
                print(f"{i}: {device['name']} (inputs: {device['max_input_channels']}, outputs: {device['max_output_channels']})")
            print("===============================\n")
        except Exception as e:
            print(f"Error listing audio devices: {e}")
    
    def transcribe_with_elevenlabs(self, audio_file: str) -> str:
        """
        Transcribe audio file using ElevenLabs REST API directly
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            return ""
        
        print(f"Transcribing audio file: {audio_file} with ElevenLabs")
        
        try:
            # ElevenLabs Speech-to-Text API endpoint (from official docs)
            url = "https://api.elevenlabs.io/v1/speech-to-text"
            
            # Headers with authentication
            headers = {
                "xi-api-key": self.api_key,
                "Accept": "application/json"
            }
            
            # Optional parameters
            params = {
                "model_id": "scribe_v1",
                "tier": "highest_accuracy",  # Prioritize accuracy over speed
                "enable_speaker_diarization": False,  # No need to identify multiple speakers
                "enable_audio_event_tagging": False,  # No need for non-verbal audio events
                "enable_language_detection": True     # Detect language automatically
            }
            
            # Open file in binary mode for multipart form upload
            with open(audio_file, "rb") as audio:
                # Create multipart form data
                files = {"file": (os.path.basename(audio_file), audio, "audio/wav")}
                
                # Make the API request
                response = requests.post(url, headers=headers, files=files, data=params)
            
            # Check response
            if response.status_code == 200:
                response_data = response.json()
                print(f"API Response: {response_data}")
                
                # Extract transcribed text
                text = response_data.get("text", "")
                print(f"Transcription: {text}")
                return text
            else:
                print(f"Error: {response.status_code}, {response.text}")
                return ""
        except Exception as e:
            print(f"Exception during transcription: {e}")
            # Print more detailed error information
            import traceback
            traceback.print_exc()
            return ""
    
    def capture_and_transcribe(self, duration: int = 5) -> str:
        """
        Record audio and transcribe in one step
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Transcribed text
        """
        audio_file = self.record_audio_using_sounddevice(duration)
        if audio_file:
            text = self.transcribe_with_elevenlabs(audio_file)
            # Clean up
            os.remove(audio_file)
            return text
        return ""
    
    def cleanup(self):
        """Remove temporary directory and files"""
        import shutil
        shutil.rmtree(self.temp_dir)
        print("Temporary files cleaned up")


# Example usage
if __name__ == "__main__":
    # Initialize the STT module (will use API key from .env file)
    stt = SpeechToText()
    
    try:
        # List available audio devices
        stt.list_audio_devices()
        
        # Set a specific input device if needed
        # Uncomment and adjust with your preferred microphone index
        # sd.default.device = [2, None]  # Use device #2 (e.g., NVIDIA Broadcast)
        
        # Test recording and transcribing
        print("Recording and transcribing...")
        text = stt.capture_and_transcribe(duration=5)
        print(f"Final transcription: {text}")
    finally:
        stt.cleanup()