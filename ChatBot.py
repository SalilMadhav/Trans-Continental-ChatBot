import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os
import requests
import tkinter as tk
from tkinter import messagebox, scrolledtext
import re

# Load environment variables
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
MODEL_NAME = "microsoft/DialoGPT-medium"

# Load the model and tokenizer
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# Perform a Google Search using SerpAPI
def search_google(query):
    if not SERPAPI_KEY:
        return "🔑 Google search unavailable. Missing API key."
    try:
        response = requests.get("https://serpapi.com/search", params={
            "q": query,
            "api_key": SERPAPI_KEY,
            "engine": "google"
        })
        data = response.json()
        result_parts = []

        if ab := data.get("answer_box", {}):
            result_parts.append(ab.get("answer") or ab.get("snippet") or "")
            result_parts.append(", ".join(ab.get("highlighted_words", [])))

        if org := data.get("organic_results", []):
            first = org[0]
            result_parts.append(f"📌 {first.get('snippet', '')}")
            result_parts.append(f"🔗 {first.get('link', '')}")

        if related := data.get("related_questions", []):
            result_parts.append("\n❓ People also ask:")
            for q in related[:3]:
                result_parts.append(f"- {q['question']}: {q.get('snippet', 'No answer found.')}")

        return "\n".join(part for part in result_parts if part).strip() or "⚠️ No useful result found."
    except Exception as e:
        return f"❌ Search failed: {str(e)}"

# GUI Application Class
class ChatBotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Smart Chatbot Assistant")
        self.root.geometry("650x700")
        self.root.config(bg="#f7f9fb")

        self.tokenizer, self.model, self.device = load_model_and_tokenizer(MODEL_NAME)
        self.chat_history_ids = None

        self.build_gui()

    def build_gui(self):
        tk.Label(self.root, text="💬 Chat with Your AI Assistant", font=("Helvetica", 18, "bold"),
                 bg="#f7f9fb", fg="#2c3e50").pack(pady=10)

        self.chat_display = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, font=("Consolas", 12),
                                                      bg="white", fg="#2c3e50", state="disabled", height=25)
        self.chat_display.pack(padx=15, pady=(0, 15), fill=tk.BOTH, expand=True)

        bottom = tk.Frame(self.root, bg="#f7f9fb")
        bottom.pack(fill=tk.X, padx=15, pady=10)

        self.user_input = tk.Entry(bottom, font=("Arial", 13), width=50)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", lambda e: self.process_input())

        send_btn = tk.Button(bottom, text="Send", font=("Arial", 12, "bold"),
                             bg="#3498db", fg="white", width=10, command=self.process_input)
        send_btn.pack(side=tk.LEFT)

        tk.Button(self.root, text="Exit", font=("Arial", 12), bg="#e74c3c", fg="white",
                  command=self.root.quit).pack(pady=(0, 15))

    def display_message(self, sender, message):
        self.chat_display.config(state="normal")
        self.chat_display.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_display.config(state="disabled")
        self.chat_display.see(tk.END)

    def process_input(self):
        user_text = self.user_input.get().strip()
        if not user_text:
            return

        self.display_message("You", user_text)
        self.user_input.delete(0, tk.END)

        if user_text.lower().startswith("google:"):
            result = search_google(user_text[7:].strip())
            self.display_message("Bot", result)
            return

        if re.fullmatch(r"[0-9+\-*/%.()\s]+", user_text):
            try:
                result = eval(user_text)
                self.display_message("Bot", f"🧮 Result: {result}")
            except Exception as e:
                self.display_message("Bot", f"❌ Error in calculation: {str(e)}")
            return

        # Generate AI response
        new_input_ids = self.tokenizer.encode(user_text + self.tokenizer.eos_token, return_tensors="pt").to(self.device)
        bot_input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1) if self.chat_history_ids is not None else new_input_ids

        self.chat_history_ids = self.model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )

        response = self.tokenizer.decode(self.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        self.display_message("Bot", response)

# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatBotApp(root)
    root.mainloop()