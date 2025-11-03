# Batman AI Chat Assistant

An interactive chat interface where users can converse with an AI-powered Batman using Google's Gemini model and Gradio.

## 🦇 Features
- Chat with Batman in his signature dark and brooding style
- Custom Batman avatar for authentic feel
- Conversation memory to maintain context
- Clean and intuitive Gradio interface
- Auto-generated responses based on Batman's perspective

## 🛠️ Setup
1. Clone the repository:
   ```powershell
   git clone https://github.com/your-username/batman-ai-chat.git
   cd batman-ai-chat
   ```

2. Install required packages:
   ```powershell
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory:
   ```plaintext
   API_KEY=your_google_api_key_here
   ```

4. Run the application:
   ```powershell
   python ai.py
   ```

## 📦 Requirements
- Python 3.7+
- langchain-google-genai
- python-dotenv
- gradio

## 🔒 Security Note
- Never commit your `.env` file
- Keep your API keys secure
- Add `.env` to your `.gitignore`

## 💬 Usage
1. Start the application
2. Open the provided local URL in your browser
3. Type your message and press Enter
4. Chat with Batman!
