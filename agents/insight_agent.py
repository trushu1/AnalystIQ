
import os
from groq import Groq
from dotenv import load_dotenv


load_dotenv()


class InsightGenerationAgent:

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please configure your .env file."
            )

        self.client = Groq(api_key=api_key)

    def generate_insights(self, analysis_report):

        prompt = f"""
        You are an expert data analyst.

        Analyze the following dataset report:

        {analysis_report}

        Provide your response in these sections:

        1. Key Findings
        2. Important Patterns
        3. Potential Data Quality Issues
        4. Business Insights
        5. Actionable Recommendations

        Use simple, clear language.
        Only report findings supported by the supplied data.
        Do not invent facts or numerical values.
        """

        response = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional AI data analyst."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3
        )

        return response.choices[0].message.content


