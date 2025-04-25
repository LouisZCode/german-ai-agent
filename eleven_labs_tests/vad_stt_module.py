import os
import tempfile
import time
import queue
import threading
import numpy as np
import requests
from typing import Optional, Tuple
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
        
        # Voice activity detection parameters - using fixed values with very low thresholds
        self.vad_threshold = 0.001  # Very low fixed threshold for reliable detection
        self.silence_limit = 2.0    # Wait longer before deciding speech is over
        self.pre_buffer = 0.5       # Keep half a second before speech detected
        self.min_speech_duration = 0.5  # Accept very short utterances
        self.debug_mode = True      # Enable debugging
    
    def record_with_vad(self, max_duration=30, sample_rate=16000, channels=1) -> Tuple[Optional[str], bool]:
        """
        Record audio with Voice Activity Detection
        - Only starts "real" recording when speech is detected
        - Automatically stops after silence is detected
        
        Args:
            max_duration: Maximum recording duration in seconds
            sample_rate: Audio sample rate
            channels: Number of audio channels
            
        Returns:
            Tuple of (path to recorded audio file, was speech detected flag)
        """
        # Create temporary file
        audio_file = os.path.join(self.temp_dir, f"recording_{int(time.time())}.wav")
        
        # Parameters
        chunk_duration = 0.1  # Process audio in 100ms chunks
        chunk_samples = int(sample_rate * chunk_duration)
        
        # Setup audio buffer
        buffer = queue.Queue()
        pre_buffer_data = []  # To store audio before speech detected
        recording_data = []   # To store the actual recording
        
        # Flags
        speech_detected = False
        recording_started = False
        last_speech_time = time.time()
        start_time = time.time()
        
        # Function to add audio to buffer
        def audio_callback(indata, frames, time_info, status):
            buffer.put(indata.copy())
        
        # Start listening
        #print("Listening for speech... (speak to start recording)")
        
        # Setup recording stream
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=channels,
            callback=audio_callback,
            dtype='float32',
            blocksize=chunk_samples
        )
        
        with stream:
            try:
                while True:
                    current_time = time.time()
                    
                    # Check for timeout
                    if current_time - start_time > max_duration:
                        print("Maximum recording duration reached")
                        break
                    
                    # Check for silence timeout after speech was detected
                    if speech_detected and current_time - last_speech_time > self.silence_limit:
                        print("Silence detected - stopping recording")
                        break
                    
                    # Get audio chunk
                    try:
                        chunk = buffer.get(timeout=1)
                    except queue.Empty:
                        continue
                    
                    # Check audio level for speech detection
                    volume_norm = np.linalg.norm(chunk) / chunk.size
                    
                    # Visual volume meter to help debug
                    if self.debug_mode:
                        meter = "Volume: " + "#" * int(volume_norm * 10000) + f" [{volume_norm:.6f}]"
                        print(meter, end="\r")
                    
                    if not speech_detected:
                        # Add to pre-buffer
                        pre_buffer_data.append(chunk)
                        
                        # Keep pre-buffer limited to specified duration
                        if len(pre_buffer_data) > int(self.pre_buffer / chunk_duration):
                            pre_buffer_data.pop(0)
                        
                        # Check if speech started
                        if volume_norm > self.vad_threshold:
                            speech_detected = True
                            recording_started = True
                            last_speech_time = current_time
                            #print("\nSpeech detected! Recording...")
                            
                            # Add pre-buffer to recording
                            recording_data.extend(pre_buffer_data)
                            recording_data.append(chunk)
                    else:
                        # Add chunk to recording
                        recording_data.append(chunk)
                        
                        # Update speech time if we hear something
                        if volume_norm > self.vad_threshold:
                            last_speech_time = current_time
                
                # Check if we recorded anything meaningful
                recording_duration = len(recording_data) * chunk_duration
                
                if not recording_started:
                    print("No speech detected")
                    return None, False
                    
                if recording_duration < self.min_speech_duration:
                    print(f"Recording too short ({recording_duration:.1f}s), ignoring")
                    return None, True
                
                print(f"Recording complete - duration: {recording_duration:.1f}s")
                
                # Combine all chunks and save to file
                recorded_audio = np.concatenate(recording_data)
                sf.write(audio_file, recorded_audio, sample_rate)
                #print(f"Audio saved to {audio_file}")
                
                return audio_file, True
                
            except Exception as e:
                print(f"Error during VAD recording: {e}")
                import traceback
                traceback.print_exc()
                return None, False
    
    def record_fixed_duration(self, duration=5, sample_rate=16000, channels=1):
        """
        Record audio for a fixed duration - fallback method
        
        Args:
            duration: Recording duration in seconds
            sample_rate: Audio sample rate
            channels: Number of audio channels
            
        Returns:
            Path to the recorded audio file
        """
        # Create temporary file
        audio_file = os.path.join(self.temp_dir, f"recording_{int(time.time())}.wav")
        
        print(f"Recording audio for {duration} seconds...")
        
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
            #print("Recording finished")
            
            # Save as WAV file
            sf.write(audio_file, recording, sample_rate)
            #print(f"Audio saved to {audio_file}")
            
            return audio_file
        
        except Exception as e:
            print(f"Error recording audio: {e}")
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
    
    def set_device(self, device_id):
        """Set the default input device"""
        sd.default.device = [device_id, None]  # [input, output]
        print(f"Using device {device_id} for input")
    
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
        
        #print(f"Transcribing audio file: {audio_file} with ElevenLabs")
        
        try:
            # ElevenLabs Speech-to-Text API endpoint
            url = "https://api.elevenlabs.io/v1/speech-to-text"
            
            # Headers with authentication
            headers = {
                "xi-api-key": self.api_key,
                "Accept": "application/json"
            }
            
            # Data payload
            data = {
                "model_id": "scribe_v1",
                "tier": "highest_accuracy",
                "enable_speaker_diarization": False,
                "enable_audio_event_tagging": False,
                "enable_language_detection": True
            }
            
            # Open file in binary mode
            with open(audio_file, "rb") as audio:
                # Create multipart form data with file and model_id
                files = {"file": (os.path.basename(audio_file), audio, "audio/wav")}
                
                # Make the API request
                response = requests.post(url, headers=headers, files=files, data=data)
            
            # Check response
            if response.status_code == 200:
                response_data = response.json()
                #print(f"API Response: {response_data}")
                
                # Extract transcribed text
                text = response_data.get("text", "")
                #print(f"Transcription: {text}")
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
    
    def capture_and_transcribe(self) -> str:
        """
        Record audio and transcribe it - tries VAD first, falls back to fixed duration
        
        Returns:
            Transcribed text
        """
        # Try recording with VAD first
        audio_file, speech_detected = self.record_with_vad()
        
        # If no audio file but speech was detected (e.g., too short), fall back to fixed duration
        if not audio_file and speech_detected:
            print("VAD detected speech but recording was too short. Using fixed duration recording.")
            audio_file = self.record_fixed_duration()
        
        # If VAD didn't detect anything, also fall back to fixed duration
        if not audio_file:
            print("VAD failed to detect speech. Using fixed duration recording.")
            audio_file = self.record_fixed_duration()
        
        # Transcribe if we have an audio file
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
        #print("Temporary files cleaned up")


# Example usage
if __name__ == "__main__":
    # Initialize the STT module
    stt = SpeechToText()
    
    try:
        # List available audio devices
        #stt.list_audio_devices()
        
        # Set a specific input device - replace 2 with your preferred device ID
        # IMPORTANT: Uncomment the next line and set your preferred mic, e.g. NVIDIA Broadcast
        # stt.set_device(2)  # Change 2 to your preferred microphone index
        
        # Test recording with VAD and transcribing
        #print("\nTesting Voice Activity Detection...")
        #print("This will show a live volume meter - speak and see if the meter responds")
        text = stt.capture_and_transcribe()
        print(f"\U0001f600 You:\n{text}")
    finally:
        stt.cleanup()