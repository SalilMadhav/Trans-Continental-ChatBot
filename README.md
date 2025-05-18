<h1>🤖 Smart Chatbot Assistant (Python + Tkinter + GPT)</h1>
A desktop chatbot app with math solving, contextual replies, Google search using SerpAPI, and a clean, responsive GUI.

<h2>💡 Features</h2>
<li>🧠 GPT-powered conversational AI (DialoGPT-medium)</li>

<li>🧮 Built-in calculator for solving math expressions</li>

<li>🌐 Google search via SerpAPI integration (google: <your-query>)</li>

<li>🖥️ Desktop GUI built using Tkinter</li>

<li>📜 Scrollable chat window with message history</li>

<li>🧼 Clean UI with modern layout</li>

<h2>🛠️ Requirements</h2>
Python 3.8+

transformers, torch, requests, python-dotenv

SerpAPI key (for Google search)

Install dependencies using:

bash
Copy
Edit
pip install transformers torch requests python-dotenv
<h2>🔑 Setup</h2>
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/smart-chatbot-assistant.git
cd smart-chatbot-assistant
Create a .env file in the root directory:

ini
Copy
Edit
SERPAPI_KEY=your_serpapi_key_here
Run the chatbot:

bash
Copy
Edit
python chatbot_gui.py
<h2>💬 Usage</h2>
Chat Normally: Type and get AI-powered responses.

Math Solver: Type any math expression like 45 * (3 + 2).

Google Search: Start your query with google:
Example: google: who is the prime minister of UK


<h2>📁 File Structure</h2>
bash
Copy
Edit
chatbot_gui.py        # Main app script
.env                  # Environment variables (not tracked in Git)
README.md             # This file
<h2>👨‍💻 Author</h2>
Developed by Salil Kelkar 
<h2>Feel free to ⭐ star the repo if you like it!</h2>
