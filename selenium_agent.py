import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

class SeleniumAgent:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        # We use Llama3 again, but with a specific instruction to write code
        self.llm = ChatGroq(
            temperature=0,
            model_name="llama-3.3-70b-versatile",
            api_key=api_key
        )

    def generate_script(self, test_case: dict, html_content: str):
        """
        Generates a Python Selenium script for a specific test case.
        """
        
        # The Prompt: Acts as a Senior QA Engineer
        prompt = ChatPromptTemplate.from_template("""
        You are a Senior QA Automation Engineer. Write a Python Selenium script to automate the following test case.
        
        TEST CASE:
        {test_case}
        
        TARGET HTML PAGE:
        {html}
        
        INSTRUCTIONS:
        1. Return ONLY the Python code. No markdown formatting like ```python.
        2. Use 'webdriver.Chrome()' (assume chromedriver is in PATH).
        3. Use strict assertions to verify the 'Expected Result'.
        4. Look at the PROVIDED HTML to find the real IDs, Names, and Classes for selectors.
        5. Include comments explaining each step.
        6. The script should print "TEST PASSED" or "TEST FAILED" at the end.
        
        PYTHON SCRIPT:
        """)

        chain = prompt | self.llm | StrOutputParser()

        try:
            # We pass the entire HTML string so the LLM can "see" the IDs
            response = chain.invoke({
                "test_case": str(test_case),
                "html": html_content
            })
            
            # Clean up markdown if the LLM still adds it despite instructions
            clean_code = response.replace("```python", "").replace("```", "").strip()
            return {"script": clean_code}
            
        except Exception as e:
            return {"error": str(e)}