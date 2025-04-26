# test_file_content.py
import sys
sys.path.append('C:\\Users\\Admin\\Desktop\\Python_Learning\\HuggingFace_AgentsCourse\\HF_First_Agent\\05_initial_agent_Voice')
import agent_stt_module

# Print the file path and the first few lines of the file
print(f"Module file path: {agent_stt_module.__file__}")
with open(agent_stt_module.__file__, 'r') as f:
    content = f.read()
    print(f"File content (first 200 chars):\n{content[:200]}")