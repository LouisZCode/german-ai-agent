import os
import requests
import json
import tempfile
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
import subprocess

MALE_TUTOR_VOICe = "vYgD8EtL4YfB6xSEYDF7"
FEMALE_TUTOR_VOICE = "CAnOszGQnhyB980lHlQP"

class TTSModule:
    """Text-to-Speech module for voice agent using ElevenLabs API."""
    
    def __init__(self, api_key: Optional[str] = None, voice_id: str = "CAnOszGQnhyB980lHlQP"):
        """
        Initialize the TTS module.
        
        Args:
            api_key: ElevenLabs API key. If None, will try to get from environment.
            voice_id: ElevenLabs voice ID to use for synthesis. Default is a common ElevenLabs voice.
        """
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key not provided and not found in environment")
            
        self.voice_id = voice_id
        self.api_url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        self.temp_dir = tempfile.gettempdir()
        self.logger = logging.getLogger(__name__)
        
    def synthesize_speech(self, text: str, output_path: Optional[str] = None) -> Path:
        """
        Convert text to speech using ElevenLabs API.
        
        Args:
            text: The text to convert to speech
            output_path: Optional path to save the audio file. If None, saves to temp directory.
            
        Returns:
            Path to the generated audio file
        """
        if not text:
            self.logger.warning("Empty text provided, skipping synthesis")
            raise ValueError("Text cannot be empty")
            
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        try:
            self.logger.info(f"Sending text to ElevenLabs API: {text[:50]}...")
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            
            if output_path:
                output_file = Path(output_path)
            else:
                output_file = Path(self.temp_dir) / f"tts_output_{hash(text)}.mp3"
                
            with open(output_file, "wb") as f:
                f.write(response.content)
                
            self.logger.info(f"Speech synthesis successful, saved to {output_file}")
            return output_file
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error synthesizing speech: {e}")
            raise
            
    def play_audio(self, audio_path: Union[str, Path]) -> None:
        """
        Play audio file using FFmpeg.
        
        Args:
            audio_path: Path to the audio file to play
        """
        audio_path = str(audio_path)  # Convert Path to string if needed
        
        try:
            # Using FFmpeg to play audio through default audio device
            cmd = ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", audio_path]
            subprocess.run(cmd, check=True)
            self.logger.info(f"Played audio file: {audio_path}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error playing audio with FFmpeg: {e}")
            raise
            
    def speak(self, text: str) -> None:
        """
        Synthesize speech and play it immediately.
        
        Args:
            text: The text to speak
        """
        try:
            audio_path = self.synthesize_speech(text)
            self.play_audio(audio_path)
        except Exception as e:
            self.logger.error(f"Error in speak method: {e}")
            raise


# Simple test function
def test_tts_module():
    """Test the TTS module with a simple text."""
    try:
        tts = TTSModule()
        test_text = "Hello, I am your voice assistant. Can you hear me?"
        print(f"Testing TTS with text: '{test_text}'")
        tts.speak(test_text)
        print("TTS test completed successfully.")
    except Exception as e:
        print(f"Error testing TTS module: {e}")


if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run test
    test_tts_module()