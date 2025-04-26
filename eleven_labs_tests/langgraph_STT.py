"""
Voice-enabled AI agent using LangGraph with ElevenLabs for speech-to-text and Anthropic's Claude.
Uses Voice Activity Detection for more natural conversations and Text-to-Speech for responses.
"""
import os
from typing import TypedDict, List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph

# Import our speech-to-text module with VAD
from vad_stt_module import SpeechToText
# Import our new text-to-speech module
from tts_module import TTSModule
import sounddevice as sd

# Load environment variables
load_dotenv()

# Get API key from environment variables
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Define the agent state type
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]  # List of conversation messages
    input_text: Optional[str]  # Latest input from user (speech)
    response_text: Optional[str]  # Latest response from assistant (for TTS)

# Initialize speech modules
stt = SpeechToText()
tts = TTSModule()  # Initialize our new TTS module

# Define nodes
def listen_node(state: AgentState) -> AgentState:
    """
    Node that listens for user input via microphone and transcribes speech
    using Voice Activity Detection
    """
    print("\n🎤 Listening... (speak to start)")
    
    # Record and transcribe using VAD (auto-starts when speech is detected)
    transcribed_text = stt.capture_and_transcribe()
    
    if not transcribed_text:
        transcribed_text = "I couldn't understand what you said. Could you please repeat?"
    
    print(f"🔊 You:\n \"{transcribed_text}\"")
    
    # Update state with transcribed text
    return {
        "input_text": transcribed_text
    }

def process_input(state: AgentState) -> AgentState:
    """
    Process the transcribed text and add it to messages
    """
    # Add user message to the conversation history
    user_message = {"role": "user", "content": state["input_text"]}
    
    # Update messages list in state
    messages = state.get("messages", [])
    messages.append(user_message)
    
    return {"messages": messages}

def agent_node(state: AgentState) -> AgentState:
    """
    Process messages with Claude and generate a response
    """
    print("🤖 Claude thinking...")
    
    # Initialize Claude with the correct model name
    claude = ChatAnthropic(
        api_key=ANTHROPIC_API_KEY,
        model="claude-3-5-haiku-20241022"  # Using the correct model name
    )
    
    # Convert to LangChain message format
    lc_messages = []
    for msg in state["messages"]:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
    
    # Get response from Claude
    response = claude.invoke(lc_messages)
    
    # Convert response back to our format
    assistant_message = {"role": "assistant", "content": response.content}
    
    # Update messages in state
    messages = state["messages"].copy()
    messages.append(assistant_message)
    
    print(f"🤖 Claude response: \"{response.content}\"")
    
    # Return updated state with response text for TTS
    return {
        "messages": messages,
        "response_text": response.content
    }

def speak_node(state: AgentState) -> AgentState:
    """
    Convert the assistant's text response to speech and play it
    """
    if state.get("response_text"):
        print("🔊 Speaking response...")
        tts.speak(state["response_text"])
        print("✓ Done speaking")
    
    # Return state unchanged
    return {}

# Build the graph
def build_agent_graph():
    """Build and return the agent graph"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("listen", listen_node)
    workflow.add_node("process_input", process_input)
    workflow.add_node("agent", agent_node)
    workflow.add_node("speak", speak_node)  # Add new TTS node
    
    # Add edges
    workflow.add_edge("listen", "process_input")
    workflow.add_edge("process_input", "agent")
    workflow.add_edge("agent", "speak")  # Agent now goes to speak
    workflow.add_edge("speak", "listen")  # After speaking, loop back to listen
    
    # Set entry point
    workflow.set_entry_point("listen")
    
    # Compile the graph
    return workflow.compile()

def main():
    """Main function to run the voice agent"""
    print("🚀 Starting Voice-Enabled AI Agent with Claude, VAD, and TTS...")
    print("Press Ctrl+C to exit")
    
    # List available audio devices
    #stt.list_audio_devices()
    
    # Set a specific input device if needed
    # Uncomment and adjust with your preferred microphone index
    # sd.default.device = [2, None]  # Use device #2 (e.g., NVIDIA Broadcast)
    
    # Build the graph
    app = build_agent_graph()
    
    # Initialize state
    initial_state = AgentState(
        messages=[],
        input_text="",
        response_text=None
    )
    
    try:
        # Run the workflow
        for state in app.stream(initial_state):
            # You could add state inspection here for debugging
            pass
    except KeyboardInterrupt:
        print("\n👋 Exiting gracefully...")
    finally:
        # Clean up resources
        stt.cleanup()

if __name__ == "__main__":
    main()