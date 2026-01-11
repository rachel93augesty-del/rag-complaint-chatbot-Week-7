# test_chatbot.py
from datetime import datetime

class SimpleChatbot:
    def get_response(self, user_message):
        user_message = user_message.lower().strip()
        
        if "credit" in user_message:
            return f"💳 **Credit Card Analysis**\n\nFound 247 complaints about credit cards.\n\n*Generated: {datetime.now().strftime('%H:%M')}*"
        elif "loan" in user_message:
            return f"💰 **Loan Analysis**\n\nFound 156 loan complaints.\n\n*Generated: {datetime.now().strftime('%H:%M')}*"
        else:
            return f"🔍 **General Analysis**\n\nI can help analyze complaints.\n\n*Generated: {datetime.now().strftime('%H:%M')}*"

# Test the chatbot
bot = SimpleChatbot()
test_queries = ["credit card complaints", "loan issues", "hello"]

print("🤖 Testing chatbot...")
for query in test_queries:
    print(f"\nQ: {query}")
    print(f"A: {bot.get_response(query)}")