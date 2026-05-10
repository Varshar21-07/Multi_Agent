AI Study Planner Multi-Agent System - Setup Guide
Prerequisites
Python 3.9+
OpenAI API Key (or Groq API Key)
Installation
Run the following command to install the required dependencies:

bash
pip install langchain langchain-openai langgraph python-dotenv rich
Environment Setup
Create a .env file in the project root with the following content:

env
OPENAI_API_KEY=your_openai_api_key_here
# If using Groq:
# GROQ_API_KEY=your_groq_api_key_here

How it Works
Input Analyzer Agent: Takes your raw goals and extracts structured data.
Study Strategy Agent: Creates a logical sequence of topics and time allocation.
Resource Recommendation Agent: Curates specific learning materials based on the strategy.
Motivation & Productivity Agent: Injects wellness tips and break schedules.
Final Planner Agent: Compiles the final roadmap into a beautiful terminal display.
Example Interaction
text
Subject: Data Structures and Algorithms
Exam Date: 2024-06-15
Hours per day: 3
Weak Topics: Dynamic Programming, Graphs
The agents will then generate a detailed day-by-day plan with specific resources and motivational tips.
