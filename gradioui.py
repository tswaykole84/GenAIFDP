import gradio as gr
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Setup agent
nutrition_agent = Agent(
    name="nutrition_agent",
    model=Gemini(id="gemini-2.5-flash-preview-04-17"),
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

# Function to call agent
def generate_diet(name, age, gender, diet_type, location, health_issues, goals, likes, dislikes, calories):
    user_input = (
        f"My name is {name}. I am {age} years old, {gender}, {diet_type}, living in {location}. "
        f"I have the following health issues: {health_issues}. "
        f"My goal is to {goals}. I like eating {likes} but dislike {dislikes}. "
        f"My daily calorie need is {calories}. "
        "Please generate a personalized meal plan broken into breakfast, lunch, snacks, dinner. "
        "Also include a clear explanation of nutritional content and a conclusion for health advice."
    )

    # Run the agent
    result = nutrition_agent.run(user_input)

    # Ensure output is readable
    if hasattr(result, "content"):
        return result.content
    elif isinstance(result, str):
        return result
    else:
        return "⚠️ Unexpected response format. Please try again."

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🥗 Smartest AI Nutrition Assistant")
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Name")
            age = gr.Textbox(label="Age")
            gender = gr.Radio(["Male", "Female", "Other"], label="Gender")
            diet_type = gr.Radio(["Vegetarian", "Non-Vegetarian"], label="Diet Type")
            location = gr.Textbox(label="Location")
            health_issues = gr.Textbox(label="Health Issues")
            goals = gr.Textbox(label="Fitness Goals")
            likes = gr.Textbox(label="Foods You Like")
            dislikes = gr.Textbox(label="Foods You Dislike")
            calories = gr.Textbox(label="Daily Calorie Requirement")
            submit = gr.Button("Get Diet Plan")
        with gr.Column():
            output = gr.Markdown(label="🧠 Your Personalized Diet Plan")

    submit.click(
        fn=generate_diet,
        inputs=[name, age, gender, diet_type, location, health_issues, goals, likes, dislikes, calories],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
