from phi.agent import Agent
from phi.model.google import Gemini
import os
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import google.generativeai as genai

# Load  API key
load_dotenv()




# Define the nutrition assistant agent
nutrition_agent = Agent(
    name="nutrition_agent",
    model=Gemini(id="gemini-2.5-flash-preview-04-17"),
    #gemini-2.5-pro-exp-03-25
    #gemini-2.5-pro-preview-05-06
    tools=[DuckDuckGo()],

    instructions=[
        "You are the smartest AI Nutrition Assistant. ",
        "Ask the user for their name, age, gender, dietary preference (veg/non-veg), location, "
        "health complaints, fitness goals, food likes/dislikes, and calorie needs. "
        "Then provide a detailed personalized diet plan (breakfast/lunch/dinner) "
        "along with clear nutritional explanations tailored to their inputs.",
    "Always respond in well-structured markdown format using clear headings like ### Breakfast, ### Lunch, etc.",
        "Do not generate one big paragraph. Instead, use bullet points, headings, and whitespace for clarity."

    ],
    show_tool_calls=True,
    markdown=True
)
